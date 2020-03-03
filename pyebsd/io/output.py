from ..ebsd import selection_to_scandata


def save_ang_file(fname, scan, sel=None, **kwargs):
    """
    Export ScanData object scan as ang file

    Arguments
    ---------
    fname : string
        File name
    scan : ScanData object
        ScanData object
    sel : list of array of booleans
        selection
    """

    newscan = selection_to_scandata(scan, sel)

    header = newscan.header
    for i, line in enumerate(header):
        if '# NCOLS_ODD:' in line:
            header[i] = '# NCOLS_ODD: {:d}\n'.format(newscan.ncols_odd)
            continue
        if '# NCOLS_EVEN:' in line:
            header[i] = '# NCOLS_EVEN: {:d}\n'.format(newscan.ncols_even)
            continue
        if '# NROWS:' in line:
            header[i] = '# NROWS: {:d}\n'.format(newscan.nrows)
            continue

    try:
        file = open(fname, 'w')
        file.write(''.join(header))
        file.close()
        newscan.data.to_csv(fname, mode='a',
                            header=False, index=False, sep=' ',
                            float_format=kwargs.pop('float_format', '%.5f'))
    except:
        raise
    else:
        print('scandata successfully saved as "{}"'.format(fname))
