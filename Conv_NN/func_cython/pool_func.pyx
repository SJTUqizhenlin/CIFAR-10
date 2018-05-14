import numpy as np 
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def poolfw_cython(x_np):
    cdef double [:,:,:] x = x_np 
    cdef Py_ssize_t i, j, d, ii, jj
    cdef Py_ssize_t xdim0 = x_np.shape[0]
    cdef Py_ssize_t xdim1 = x_np.shape[1]
    cdef Py_ssize_t xdim2 = x_np.shape[2]
    cdef Py_ssize_t resdim0 = xdim0
    cdef Py_ssize_t resdim1 = xdim1 / 2
    cdef Py_ssize_t resdim2 = xdim2 / 2
    res_np = np.zeros((resdim0, resdim1, resdim2)) 
    cdef double [:,:,:] res = res_np 
    cdef double maxval
    for d in range(resdim0):
        for i in range(resdim1):
            for j in range(resdim2):
                ii = i * 2
                jj = j * 2
                maxval = x[d, ii, jj]
                if x[d, ii, jj + 1] > maxval:
                    maxval = x[d, ii, jj + 1]
                if x[d, ii + 1, jj] > maxval:
                    maxval = x[d, ii + 1, jj]
                if (x[d, ii + 1, jj + 1] > maxval):
                    maxval = x[d, ii + 1, jj + 1]
                res[d, i, j] = maxval 
    return res_np 

@cython.boundscheck(False)
@cython.wraparound(False)
def poolbw_cython(bg_np, x_np):
    cdef double [:,:,:] bg = bg_np 
    cdef double [:,:,:] x = x_np 
    cdef Py_ssize_t i, j, d, dx, dy, ii, jj
    cdef Py_ssize_t xdim0 = x_np.shape[0]
    cdef Py_ssize_t xdim1 = x_np.shape[1]
    cdef Py_ssize_t xdim2 = x_np.shape[2]
    cdef Py_ssize_t bgdim0 = bg_np.shape[0]
    cdef Py_ssize_t bgdim1 = bg_np.shape[1]
    cdef Py_ssize_t bgdim2 = bg_np.shape[2]
    res_np = np.zeros((xdim0, xdim1, xdim2)) 
    cdef double [:,:,:] res = res_np 
    for d in range(bgdim0):
        for i in range(bgdim1):
            for j in range(bgdim2):
                ii = i * 2
                jj = j * 2
                maxval = x[d, ii, jj]
                dx = 0
                dy = 0
                if x[d, ii, jj + 1] > maxval:
                    maxval = x[d, ii, jj + 1]
                    dx = 0
                    dy = 1
                if x[d, ii + 1, jj] > maxval:
                    maxval = x[d, ii + 1, jj]
                    dx = 1
                    dy = 0
                if (x[d, ii + 1, jj + 1] > maxval):
                    maxval = x[d, ii + 1, jj + 1]
                    dx = 1
                    dy = 1
                res[d, ii + dx, jj + dy] = bg[d, i, j]
    return res_np 