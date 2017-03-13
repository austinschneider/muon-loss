import argparse
import glob
import itertools
import os
import os.path as path
import re
import pickle

def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)

def limit_range(infiles, file_range):
    if file_range is not None:
        try:
            file_range = list(file_range)
            if len(file_range) == 1:
                infiles = [infiles[file_range]]
            elif len(file_range) >= 2:
                infiles = infiles[file_range[0]:file_range[1]]
        except:
            try:
                file_range = int(file_range)
                infiles = [infiles[file_range]]
            except:
                pass
    return infiles

def get_output_file(infiles):
    prefix = path.commonprefix(infiles)
    suffix = path.commonprefix([f[::-1] for f in infiles])[::-1]
    num_suffix = ''.join([c for c in itertools.takewhile(lambda x: x.isdigit(), suffix)])
    num_prefix = ''.join([c for c in itertools.takewhile(lambda x: x.isdigit(), prefix[::-1])])[::-1]
    prefix = prefix[:len(prefix)-len(num_prefix)]
    suffix = suffix[len(num_suffix):]
    diffs = [s[len(prefix):len(s)-len(suffix)] for s in infiles]
    output_file = prefix + min(diffs, key=alphanum_key) + '-' + max(diffs, key=alphanum_key) + suffix
    return output_file

def get_output_files(infiles, outdir, mkdir=True, file_range=None, common=None):
    outdir = path.dirname(outdir+'/')
    if type(infiles) == str:
        infiles = glob.glob(infiles)
    sort_nicely(infiles)

    if common is None:
        common = path.commonprefix(infiles)
        common = path.dirname(common)
    common_len = len(common)
    infiles = limit_range(infiles, file_range)
    structure_set = set([path.dirname(s[common_len+1:]) for s in infiles])
    total_structure_set = set()

    for s in structure_set:
        ss = s.split('/')
        total_s = outdir
        for sss in ss:
            total_s = '/'.join([total_s,sss])
            total_structure_set.add(total_s)

    output_dirs = sorted(list(total_structure_set))

    if mkdir:
        for d in output_dirs:
            if not path.exists(d):
                os.mkdir(d)

    infile_sets = dict()
    for f in infiles:
        d = path.dirname(f)
        if d not in infile_sets.keys():
            infile_sets[d] = []
        infile_sets[d].append(f)

    file_pairs = []
    for indir, infiles in infile_sets.items():
        outfile_name = path.basename(get_output_file(infiles))
        outfile = '/'.join([s for s in [outdir, indir[common_len:], outfile_name] if s != ''])
        file_pairs.append((infiles, outfile))
    return file_pairs

def get_split_output_files(infiles_per_outfile, infiles, outdir, mkdir=True, file_range=None):
    infiles = glob.glob(infiles)
    sort_nicely(infiles)
    common = path.commonprefix(infiles)
    infiles = limit_range(infiles, file_range)
    file_chunks = [infiles[i:i+infiles_per_outfile] for i in xrange(0, len(infiles), infiles_per_outfile)]
    chunk_pairs = [get_output_files(chunk, outdir, mkdir, common=common) for chunk in file_chunks]
    return chunk_pairs

def store_output_files(file_name, split_output_files):
    outfile = open(file_name, 'wb')
    pickle.dump(split_output_files, outfile, -1)

def load_output_files(file_name, i):
    infile = open(file_name, 'rb')
    return pickle.load(infile)[i]

