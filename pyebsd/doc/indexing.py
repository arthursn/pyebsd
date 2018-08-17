"""
Indexing hexagonal binning

n : ncols_odd
m : ncols_even
N : ncols_odd + ncols_even

              indices ind
     0     1     2  ...  n-2   n-1
     *     *     *        *     *
        n    n+1    ...     N-1
        *     *              *     
     N    N+1   N+2 ... N+n-2 N+n-1
     *     *     *        *     *


               columns j
     0     2     4  ...  N-3   N-1
     *     *     *        *     *
        1     3     ...     N-2      (row i)
        *     *              * 
     -----------------------------
     *     *     *        *     *
                    ...              (row i+1)
        *     *              * 

i = ind/N
j = (ind%N)/n + 2*((ind%N)%n)


"""