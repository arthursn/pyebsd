if __name__ == '__main__':
    import argparse
    from .io import load_scandata

    parser = argparse.ArgumentParser()
    parser.add_argument('file', nargs=1)
    parser.add_argument('--ipf', action='store_true')
    parser.add_argument('--pf', action='store_true')
    args = parser.parse_args()

    scan = load_scandata(args.file[0])

    if args.ipf:
        ipf = scan.plot_IPF(gray=scan.IQ)
        ipf.fig.show()

    if args.pf:
        ax = scan.plot_PF(contour=True)
        ax.get_figure().show()
