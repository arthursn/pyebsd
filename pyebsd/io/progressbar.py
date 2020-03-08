import sys

__all__ = ['ProgressBar']


class ProgressBar(object):
    def __init__(self, niterations, barsize=50, progresschar='#',
                 fillingchar='.', separatorleft='[', separatorright=']'):

        self.nit = niterations
        self.size = barsize
        self.each = float(self.nit/self.size)

        self.nprg = 0
        self.prgchr = progresschar[0]
        self.fillchr = fillingchar[0]

        self.seplft = separatorleft
        self.seprgt = separatorright

        self.seplftsize = len(self.seplft)
        self.seprgtsize = len(self.seprgt)

    def initialize(self):
        self.nprg = 0
        self.draw(0)  # draws empty progress bar

    def draw(self, it):
        nprg = int(round(it/self.each))
        # updates bar only when necessary
        if nprg > self.nprg or it + 1 == self.nit:
            self.nprg = nprg

            # progress in percentage
            prg = 100.*it/self.nit

            # number of filling chars that will be printed
            nfill = self.size - nprg + self.seprgtsize
            if nfill < 0:
                nfill = 0
            # number of progress chars that will be printed
            nprg += self.seplftsize

            # draws progressbar
            sys.stdout.write('{:4.0f}% {:{}<{}}{:{}>{}}\r'.format(
                prg, self.seplft, self.prgchr, nprg,
                self.seprgt, self.fillchr, nfill))

            # line break when finished
            if it + 1 == self.nit:
                sys.stdout.write('\n')

            sys.stdout.flush()


if __name__ == '__main__':
    import time
    n = 5839
    pbar = ProgressBar(n)
    pbar.initialize()
    for i in range(n):
        pbar.draw(i)
        time.sleep(1e-4)
