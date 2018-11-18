def save_ang_file(fname, scan, sel=None):
    """
    Export Scandata object scan as ang file

    Arguments
    ---------
    fname : string
        File name
    scan : Scandata object
        Scandata object
    sel : list of array of booleans
        selection
    """

    scan.header  # header from the original ang file
    dataexport = scan.data.copy()  # raw data
    dataexport[~sel]



    pass