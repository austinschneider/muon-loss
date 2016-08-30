import os
import time
import uuid
import shutil
import random
import threading
import collections

class buffered_task_thread(threading.Thread):
    """
    Thread to perform work on data in buffer
    """
    def __init__(self, task, sleep_period=0.1):
        threading.Thread.__init__(self)
        self.task = task
        self.data = []
        self.lock = threading.Lock()
        self.sleep_period = sleep_period
        self.stop = False
    def run(self):
        while True:
            has_next = False
            self.lock.acquire()
            if len(self.data) > 0:
                my_data = self.data
                self.data = []
                has_next = True
            self.lock.release()
            if has_next:
                print len(my_data)
                for obj in my_data:
                    self.task(obj)
            elif self.stop:
                return
            else:
                time.sleep(self.sleep_period)

class buffered_task:
    """
    Class to encapsulate thread instance and handle locking on data
    """
    def __init__(self, task, sleep_period=0.1, max_data=None):
        self.thread = buffered_task_thread(task, sleep_period=sleep_period)
        self.max_data = max_data
        self.sleep_period = sleep_period
        self.thread.start()
    def add(self, obj):
        while True:
            self.thread.lock.acquire()
            n = len(self.thread.data)
            self.thread.lock.release()
            if self.max_data is None or n < self.max_data:
                self.thread.lock.acquire()
                self.thread.data.append(obj)
                self.thread.lock.release()
                return
            else:
                time.sleep(self.sleep_period)

    def join(self):
        self.thread.stop = True
        self.thread.join()

class file_writer_thread(threading.Thread):
    """
    Thread for writing data to file
    Write method should accept list of data
    """
    def __init__(self, file, write, sleep_period=0.1):
        threading.Thread.__init__(self)
        self.file = file
        self.write = write
        self.buffer = []
        self.lock = threading.Lock()
        self.sleep_period = sleep_period
        self.stop = False

    def run(self):
        while True:
            has_next = False
            self.lock.acquire()
            if len(self.buffer) > 0:
                my_queue = self.buffer
                self.buffer = []
                has_next = True
            self.lock.release()
            if has_next:
                print "Writing %d objects" % len(my_queue)
                self.write(self.file, my_queue)
            elif self.stop:
                return
            time.sleep(self.sleep_period)

class file_reader_thread(threading.Thread):
    """
    Thread for reading data from file
    Read method should return list of data
    """
    def __init__(self, file, read, sleep_period=0.1, buffer_max = 100000):
        threading.Thread.__init__(self)
        self.file = file
        self.read = read
        self.buffer = []
        self.lock = threading.Lock()
        self.sleep_period = sleep_period
        self.stop = False
        self.buffer_max = buffer_max

    def run(self):
        while True:
            self.lock.acquire()
            n = len(self.buffer)
            self.lock.release()
            if n < self.buffer_max:
                n_to_get = self.buffer_max - n
                temp_buffer = []
                while len(temp_buffer) < n_to_get or self.stop:
                    try:
                        temp_buffer += self.read(self.file)
                    except:
                        self.stop = True
                self.lock.acquire()
                buffer += temp_buffer
                del temp_buffer
                self.lock.release()
            else:
                time.sleep(self.sleep_period)
            if self.stop:
                return

class file_writer:
    """
    Threaded file writer
    Class to encapsulate thread instance and handle locking on data
    Write method should accept list of data
    """
    def __init__(self, file, write, sleep_period=0.1):
        self.thread = file_writer_thread(file, write, sleep_period)
        self.thread.start()
    def write(self, obj):
        self.thread.lock.acquire()
        self.thread.buffer.append(obj)
        self.thread.lock.release()
    def join(self):
        self.thread.stop = True
        self.thread.join()

class file_reader:
    """
    Threaded file reader
    Class to encapsulate thread instance and handle locking on data
    Read method should return list of data
    """
    def __init__(self, file, read, sleep_period=0.1, max_buffer=100000):
        self.thread = file_reader_thread(file, read, sleep_period, max_buffer)
        self.sleep_period = sleep_period
        self.buffer = []
        self.n = 0
        self.thread.start()
    def __iter__(self):
        return self
    def read(self):
        while len(self.buffer) == 0:
            self.thread.lock.acquire()
            if len(self.thread.buffer) > 0:
                self.buffer = self.thread.buffer
                self.thread.buffer = []
            elif self.thread.stop:
                self.thread.lock.release()
                raise ValueError('No items left')
            else:
                self.thread.lock.release()
                time.sleep(self.sleep_period)
                continue
            self.thread.lock.release()
            self.buffer.reverse()
        self.n += 1
        return self.buffer.pop()
    def next(self):
        try:
            return self.read()
        except:
            print "Read %d items!" % self.n
            raise StopIteration()

class scratch_writer:
    """
    Threaded file writer that writes to scratch and then copies to destination
    Class to encapsulate thread instance and handle locking on data
    Write method should accept list of data
    """
    def __init__(self, file_name, write, sleep_period=0.1):
        scratch_dir = '/scratch/%s/' % os.environ['USER']
        if not os.path.exists(scratch_dir):
            os.makedir(scratch_dir)
        self.scratch_file_name = scratch_dir + str(uuid.uuid4())
        self.scratch_file = open(self.scratch_file_name, 'w')
        self.writer = file_writer(self.scratch_file, write, sleep_period)
        self.file_name = file_name
    def write(self, obj):
        self.writer.write(obj)
    def join(self):
        self.writer.join()
        self.scratch_file.close()
        shutil.copy(self.scratch_file_name, self.file_name)
        os.remove(self.scratch_file_name)

class scratch_reader:
    """
    Threaded file reader that copies file to scratch before reading
    Class to encapsulate thread instance and handle locking on data
    Read method should return list of data
    """
    def __init__(self, file_name, read, sleep_period=0.1, max_buffer=100000):
        scratch_dir = '/scratch/%s/' % os.environ['USER']
        if not os.path.exists(scratch_dir):
            os.makedir(scratch_dir)
        self.scratch_file_name = scratch_dir + str(uuid.uuid4())
        self.file_name = file_name
        shutil.copy(self.file_name, self.scratch_file_name)
        self.scratch_file = open(self.scratch_file_name, 'r')
        self.reader = file_reader(self.scratch_file, read, sleep_period, max_buffer)
    def __iter__(self):
        return self
    def read(self):
        return self.reader.read()
    def next(self):
        try:
            return self.reader.next()
        except:
            raise StopIteration()
    def join(self):
        self.writer.join()
        self.scratch_file.close()
        os.remove(self.scratch_file_name)

def write_generic(file, obj):
    file.write(obj)
