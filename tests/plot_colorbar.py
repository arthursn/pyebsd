# -*- coding: utf-8 -*-

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--vmin', type=float, default=0)
    parser.add_argument('-M', '--vmax', type=float, default=5)
    parser.add_argument('-n', '--levels', type=float, default=None)
    
    parser.add_argument('-c', '--cmap', default='viridis')
    parser.add_argument('-l', '--label', default=u'Deviation from KS OR (°)')
    
    parser.add_argument('-f', '--figsize', type=float,
                        nargs=2, default=[6, .5])
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=args.figsize)

    # Set the colormap and norm to correspond to the data for which
    # the colorbar will be used.
    cmap = plt.get_cmap(args.cmap)
    if args.levels:
        # extract all colors from the .jet map
        # cmaplist = [cmap(i) for i in range(cmap.N)]
        # force the first color entry to be grey
        # cmaplist[0] = (.5,.5,.5,1.0)
        # create the new map
        # cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
        norm = mpl.colors.BoundaryNorm(np.linspace(args.vmin, args.vmax, args.levels), cmap.N)
    else:
        norm = mpl.colors.Normalize(vmin=args.vmin, vmax=args.vmax)

    # ColorbarBase derives from ScalarMappable and puts a colorbar
    # in a specified axes, so it has everything needed for a
    # standalone colorbar.  There are many more kwargs, but the
    # following gives a basic continuous colorbar with ticks
    # and labels.
    cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                   norm=norm,
                                   orientation='horizontal')
    cb.set_label(args.label)

    fname = 'colobar_{:.0f}-{:.0f}_deg.pdf'.format(args.vmin, args.vmax)
    
    fig.savefig(fname, bbox_inches='tight', pad_inches=0)
