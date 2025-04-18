from fealpy.backend import backend_manager as bm

a = bm.ones(shape=3, dtype=bm.complex128)
print(a, type(a))