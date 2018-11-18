import sys
import time
import numpy as np
import pandas as pd


def load_ang_file(fname):
    t0 = time.time()
    print('Reading file \"{}\"...'.format(fname))
    
    # Read and parse header
    with open(fname) as f:
        header = ''
        nmatches = 0
        for line in f:
            pattern = '# GRID:'
            if pattern in line.upper():
                grid = line.split(pattern)[-1].strip()
                print(line.strip())
                nmatches += 1
                continue
            pattern = '# XSTEP:'
            if pattern in line.upper():
                dx = float(line.split(pattern)[-1])
                print(line.strip())
                nmatches += 1
                continue
            pattern = '# YSTEP:'
            if pattern in line.upper():
                dy = float(line.split(pattern)[-1])
                print(line.strip())
                nmatches += 1
                continue
            pattern = '# NCOLS_ODD:'
            if pattern in line.upper():
                ncols_odd = int(line.split(pattern)[-1])
                print(line.strip())
                nmatches += 1
                continue
            pattern = '# NCOLS_EVEN:'
            if pattern in line.upper():
                ncols_even = int(line.split(pattern)[-1])
                print(line.strip())
                nmatches += 1
                continue
            pattern = '# NROWS:'
            if pattern in line.upper():
                nrows = int(line.split(pattern)[-1])
                print(line.strip())
                nmatches += 1
                continue
            if nmatches == 6:
                break

            header += line

    # Uses pandas to read ang file. pd.read_table returns a pandas DataFrame
    # .values converts the DataFrame into a numpy array
    data = pd.read_table(fname, header=None, comment='#',
                         delim_whitespace=True).values

    print('\n{} points read in {:.2f} s'.format(len(data), time.time() - t0))

    return data, grid, dx, dy, ncols_odd, ncols_even, nrows, header
