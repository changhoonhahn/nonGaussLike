'''
'''
import ctypes
import numpy as np


lib_z1 = ctypes.CDLL("Ccode/library_combined_win_local_z1.so")
    

def taruya_model(mubins, binrange1, binrange2, binrange3, maxbin1, x, alpha_perp, alpha_para, fsig8, b1NGCsig8, b1SGCsig8, b2NGCsig8, b2SGCsig8, NNGC, NSGC, sigmavNGC, sigmavSGC):
    ''' Python wrapper for Taruya model in Florian's paper 
    '''
    x_arr = (ctypes.c_double * len(x))(*x)
    lib.taruya_model_combined_win_local.restype = ctypes.POINTER(ctypes.c_double * len(x)*2)
    
    result = lib.taruya_model_combined_win_local(ctypes.c_int(mubins),
            ctypes.c_int(binrange1),ctypes.c_int(binrange2),ctypes.c_int(binrange3), ctypes.c_int(maxbin1),
            x_arr,ctypes.c_double(alpha_perp),ctypes.c_double(alpha_para),
            ctypes.c_double(fsig8),ctypes.c_double(b1NGCsig8),ctypes.c_double(b1SGCsig8),ctypes.c_double(b2NGCsig8),ctypes.c_double(b2SGCsig8),
            ctypes.c_double(NNGC), ctypes.c_double(NSGC),ctypes.c_double(sigmavNGC),ctypes.c_double(sigmavSGC))
    model = [i for i in result.contents]
    model = np.array(model)
    return model
