import sys, time
import numpy as np
from ..draw import ProgressBar

def load_ang_file(fname):
    t0 = time.time()
    print('Reading file \"{}\"'.format(fname))
    with open(fname) as f:
        matches = 0
        for line in f:
            pattern = '# GRID:'
            if pattern in line.upper():
                grid = line.split(pattern)[-1].strip()
                print(line.strip())
                matches += 1
            pattern = '# XSTEP:'
            if pattern in line.upper():
                dx = float(line.split(pattern)[-1])
                print(line.strip())
                matches += 1
            pattern = '# YSTEP:'
            if pattern in line.upper():
                dy = float(line.split(pattern)[-1])
                print(line.strip())
                matches += 1
            pattern = '# NCOLS_ODD:'
            if pattern in line.upper():
                ncols_odd = int(line.split(pattern)[-1])
                print(line.strip())
                matches += 1
            pattern = '# NCOLS_EVEN:'
            if pattern in line.upper():
                ncols_even = int(line.split(pattern)[-1])
                print(line.strip())
                matches += 1
            pattern = '# NROWS:'
            if pattern in line.upper():
                nrows = int(line.split(pattern)[-1])
                print(line.strip())
                matches += 1
            if matches == 6:
                break

        nrow, ncol = np.int(np.ceil(nrows/2.)*ncols_odd + np.floor(nrows/2.)*ncols_even), 14
        data = np.empty((nrow, ncol), dtype = np.double)
        
        prgbar = ProgressBar(nrow)  # progress bar
        prgbar.initialize() # draw empty progress bar

        irow = 0
        for line in f:
            if line[0] == '#' or line == '\n':
                continue
            else:
                for icol, s in enumerate(line.split()):
                    data[irow, icol] = float(s)
                irow += 1
                prgbar.draw(irow)  # updates progress bar status
            if irow >= nrow:
                break

    print('\n{} points read in {:.2f} s'.format(irow, time.time() - t0))

    return data, grid, dx, dy, ncols_odd, ncols_even, nrows
