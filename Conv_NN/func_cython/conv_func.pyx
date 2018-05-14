import numpy as np 
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def convfw_cython(x_np, k_np):
    cdef double [:,:,:] x = x_np 
    cdef double [:,:,:] k = k_np 
    cdef Py_ssize_t p, q, i, j, d
    cdef Py_ssize_t xdim0 = x_np.shape[0]
    cdef Py_ssize_t xdim1 = x_np.shape[1]
    cdef Py_ssize_t xdim2 = x_np.shape[2]
    cdef Py_ssize_t kdim0 = k_np.shape[0]
    cdef Py_ssize_t kdim1 = k_np.shape[1]
    cdef Py_ssize_t kdim2 = k_np.shape[2]
    cdef Py_ssize_t resdim1 = xdim1 - kdim1 + 1
    cdef Py_ssize_t resdim2 = xdim2 - kdim2 + 1
    res_np = np.zeros((resdim1, resdim2)) 
    cdef double [:,:] res = res_np 
    cdef double val
    for p in range(resdim1):
        for q in range(resdim2):
            val = 0.0
            for d in range(xdim0):
                for i in range(kdim1):
                    for j in range(kdim2):
                        val += x[d, p + i, q + j] * k[d, i, j]
            res[p, q] = val 
    return res_np 

@cython.boundscheck(False)
@cython.wraparound(False)
def convbw_cython(bg_np, x_np, k_np):
    cdef double [:,:] bg = bg_np 
    cdef double [:,:,:] x = x_np 
    cdef double [:,:,:] k = k_np 
    cdef Py_ssize_t p, q, i, j, d
    cdef Py_ssize_t xdim0 = x_np.shape[0]
    cdef Py_ssize_t xdim1 = x_np.shape[1]
    cdef Py_ssize_t xdim2 = x_np.shape[2]
    cdef Py_ssize_t kdim0 = k_np.shape[0]
    cdef Py_ssize_t kdim1 = k_np.shape[1]
    cdef Py_ssize_t kdim2 = k_np.shape[2]
    cdef Py_ssize_t bgdim1 = bg_np.shape[0]
    cdef Py_ssize_t bgdim2 = bg_np.shape[1]
    res1_np = np.zeros((xdim0, xdim1, xdim2))
    res2_np = np.zeros((kdim0, kdim1, kdim2))
    cdef double [:,:,:] res1 = res1_np
    cdef double [:,:,:] res2 = res2_np 
    cdef double tmpbg
    for p in range(bgdim1):
        for q in range(bgdim2):
            tmpbg = bg[p, q]
            for d in range(kdim0):
                for i in range(kdim1):
                    for j in range(kdim2):
                        res1[d, p + i, q + j] += tmpbg * k[d, i, j]
                        res2[d, i, j] += tmpbg * x[d, p + i, q + j]
    cdef Py_ssize_t n = bgdim1 * bgdim2 
    for d in range(kdim0):
        for i in range(kdim1):
            for j in range(kdim2):
                res2[d, i, j] /=  n 
    return res1_np, res2_np 