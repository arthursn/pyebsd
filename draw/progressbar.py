import sys

class ProgressBar(object):
    def __init__(self, niterations, barsize=50, 
                 progresschar='#', fillingchar='.', separatorleft='[', separatorright=']'):

        self.nit = niterations
        self.size = barsize
        self.each = float(self.nit/self.size)

        self.nprg = 0
        self.prgchr = progresschar[0]
        self.fillchr = fillingchar[0]

        self.seplft = separatorleft
        self.seprgt = separatorright

    def initialize(self):
        self.nprg = 0
        self.draw(0)  # draws empty progress bar

    def draw(self, it):
        if it/self.each >= self.nprg:
            prg = 100.*it/self.nit  # progress in percentage

            nprg = self.nprg + len(self.seplft)  # number of progress chars that will be printed
            nfill = self.size - self.nprg + len(self.seprgt)  # number of filling chars that will be printed

            sys.stdout.write('{:4.0f}% {:{}<{}}{:{}>{}}\r'.format(prg, self.seplft, self.prgchr, nprg, self.seprgt, self.fillchr, nfill))  # draws progressbar
            sys.stdout.flush()

            self.nprg += 1

            if self.nprg > self.size:
                sys.stdout.write('\n')
