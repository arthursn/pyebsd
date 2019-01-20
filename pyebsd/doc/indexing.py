"""
Indexing hexagonal tiling

n : ncols_odd
m : ncols_even
N : ncols_odd + ncols_even


              indices ind
     0     1     2  ...  n-2   n-1
     *     *     *        *     *
        n    n+1    ...     n+m
        *     *              *     
    n+m  n+m+1      ...
     *     *     *        *     *
                     .
                     .
                     .

        *     *     ...      *

     *     *     *  ...   *     *


               columns j
     0     2     4  ...  N-3   N-1
     *     *     *        *     *   (row 0)
        1     3     ...     N-2
        *     *              *      (row 1)

     *     *     *  ...   *     *   (row 2)
                     .
                     .
                     .

        *     *     ...      *      (row i)

     *     *     *  ...   *     *   (row i+1)

"""