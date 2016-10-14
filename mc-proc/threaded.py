import os
import copy
import time
import uuid
import shutil
import random
import threading
import collections
import multiprocessing
import Queue
class sleep_time:
    def __init__(self, min_sleep, max_sleep, history_length=10):
        self.history = collections.deque([min_sleep for i in xrange(history_length)])
        self.min = min_sleep
        self.max = max_sleep
        self.avg = min_sleep
        self.fresh = False
        self.n = history_length
    def sleep(self):
        if self.fresh:
            t = self.avg/4.0
            self.fresh = False
        else:
            t = self.history.pop()
            self.history.append(t)
        
        tt = self.history.popleft()
        self.avg = self.avg - tt/self.n + t*2.0/self.n
        tt = min(max(t*2.0, self.min), self.max)
        self.history.append(tt)
        time.sleep(tt)
    def reset(self):
        self.fresh = True


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
                #print len(my_data)
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

class file_writer_thread(multiprocessing.Process):
    """
    Thread for writing data to file
    Write method should accept list of data
    """
    def __init__(self, file, write, sleep_period=0.1, buffer_max = 100000, chunk = 100):
        multiprocessing.Process.__init__(self)
        self.file = file
        self.write = write
        self.queue = multiprocessing.Queue(buffer_max)
        self.buffer_max = buffer_max
        self.chunk = chunk
        self.sleep_period = sleep_period

    def run(self):
        while True:
            try:
                elems = self.queue.get(True, self.sleep_period)
                if elems is Queue.Empty:
                    print 'Ending writer thread'
                    self.file.close()
                    return
                elems.reverse()
                self.write(self.file, elems)
            except Queue.Empty as e:
                print 'Empty'
                pass
            except Exception as e:
                print 'Got other exception'
                print e
                raise
    def stop(self):
        self.queue.put(Queue.Empty)

class file_reader_thread(multiprocessing.Process):
    """
    Thread for reading data from file
    Read method should return list of data
    """
    def __init__(self, file, read, sleep_period=0.1, buffer_max = 100000, chunk=100):
        multiprocessing.Process.__init__(self)
        self.file = file
        self.read = read
        self.queue = multiprocessing.Queue(buffer_max)
        self.stop = False
        self.buffer_max = buffer_max
        self.chunk = chunk

    def run(self):
        while True:
            try:
                print 'Trying to read'
                elems = self.read(self.file)
                print 'Read'
                while len(elems) > 0:
                    print 'Trying to put'
                    self.queue.put(elems[:self.chunk])
                    print 'Put'
                    elems = elems[self.chunk:]
            except Exception as e:
                print 'Got exception: %s' % str(e)
                print 'Issue reading more from file'
                self.stop = True
                self.queue.put(Queue.Empty)
            if self.stop:
                print 'Ending reader thread'
                return

class file_writer:
    """
    Threaded file writer
    Class to encapsulate thread instance and handle locking on data
    Write method should accept list of data
    """
    def __init__(self, file_name, write, sleep_period=0.1, buffer_max=100000, chunk=100):
        file = open(file_name, 'w')
        self.file_name = file_name
        self.thread = file_writer_thread(file, write, sleep_period, buffer_max, chunk)
        self.buffer = []
        self.chunk = chunk
        self.thread.start()
    def write(self, obj):
        self.buffer.append(obj)
        if len(self.buffer) >= self.chunk:
            self.thread.queue.put(self.buffer)
            self.buffer = []
    def join(self):
        if len(self.buffer) > 0:
            self.thread.queue.put(self.buffer)
            self.buffer = []
        self.thread.stop()
        self.thread.join()
        self.thread.file.close()
    def restart(self):
        self.thread.stop()
        self.thread.join()
        self.thread.file.close()
        file = open(self.file_name, 'a')
        self.thread = file_writer_thread(file, self.thread.write, self.thread.sleep_period, self.thread.buffer_max, self.thread.chunk)
        self.thread.start()

class file_reader:
    """
    Threaded file reader
    Class to encapsulate thread instance and handle locking on data
    Read method should return list of data
    """
    def __init__(self, file, read, sleep_period=0.1, max_buffer=100, chunk=100):
        self.thread = file_reader_thread(file, read, sleep_period, max_buffer, chunk)
        self.sleep_period = sleep_period
        self.n = 0
        self.thread.start()
        self.elems = []
    def __iter__(self):
        return self
    def read(self):
        if len(self.elems) > 0:
            print 'Returning element'
            return self.elems.pop()
        while True:
            try:
                self.elems = self.thread.queue.get(True, self.sleep_period)
                if self.elems is Queue.Empty:
                    self.thread.queue.put(Queue.Empty)
                    print 'Ending reader'
                    raise StopIteration()
                self.elems.reverse()
                print 'Got an element'
                print 'Returning elemet'
                return self.elems.pop()
            except Queue.Empty as e:
                print 'Empty'
                pass
    def next(self):
        try:
            res = self.read()
        except StopIteration as e:
            print 'Got StopIteration'
            print e
            print "Read %d items!" % self.n
            raise e
        except Exception as e:
            print 'Got other exception'
            print e
            print "Read %d items!" % self.n
            raise e
        return res
    def join(self):
        #self.thread.stop = True
        self.thread.join()
        self.thread.file.close()
    def copy(self):
        c = copy.copy(self)
        c.sleep = sleep_time(c.sleep_period / 10.0, c.sleep_period)
        c.buffer = collections.deque([])
        return c

class scratch_writer:
    """
    Threaded file writer that writes to scratch and then copies to destination
    Class to encapsulate thread instance and handle locking on data
    Write method should accept list of data
    """
    def __init__(self, file_name, write, sleep_period=0.01):
        scratch_dir = '/scratch/%s/' % os.environ['USER']
        if not os.path.exists(scratch_dir):
            os.makedir(scratch_dir)
        self.scratch_file_name = scratch_dir + str(uuid.uuid4())
        #self.scratch_file = open(self.scratch_file_name, 'w')
        self.writer = file_writer(self.scratch_file_name, write, sleep_period)
        self.thread = self.writer.thread
        self.file_name = file_name
    def write(self, obj):
        self.writer.write(obj)
    def join(self):
        self.writer.join()
        shutil.copy(self.scratch_file_name, self.file_name)
        os.remove(self.scratch_file_name)
    def restart(self):
        self.writer.restart()
        self.thread = self.writer.thread

class scratch_reader:
    """
    Threaded file reader that copies file to scratch before reading
    Class to encapsulate thread instance and handle locking on data
    Read method should return list of data
    """
    def __init__(self, file_name, read, sleep_period=0.01, max_buffer=1000):
        scratch_dir = '/scratch/%s/' % os.environ['USER']
        if not os.path.exists(scratch_dir):
            os.makedir(scratch_dir)
        self.scratch_file_name = scratch_dir + str(uuid.uuid4())
        self.file_name = file_name
        print shutil.copy(self.file_name, self.scratch_file_name)
        self.scratch_file = open(self.scratch_file_name, 'r')
        self.reader = file_reader(self.scratch_file, read, sleep_period, max_buffer)
    def __iter__(self):
        return self
    def read(self):
        return self.reader.read()
    def next(self):
        return self.reader.next()
    def join(self):
        self.reader.join()
        self.scratch_file.close()
        try:
            os.remove(self.scratch_file_name)
        except:
            pass
    def copy(self):
        c = copy.copy(self)
        c.reader = self.reader.copy()
        return c

def write_generic(file, obj):
    file.write(obj)
