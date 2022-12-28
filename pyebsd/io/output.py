__all__ = ["save_ang_file"]


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

    scan.save_ang_file(fname, sel, **kwargs)
