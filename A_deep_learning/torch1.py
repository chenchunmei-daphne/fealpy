from fealpy.backend import backend_manager as bm
bm.set_backend('pytorch')
from fealpy.ml.sampler import ISampler

domain = (-1, 1, -1, 1)
in_sampler = ISampler(ranges=domain, mode='linspace', 
                      dtype=bm.float64, device=None, 
                      requires_grad=False)
in_point = in_sampler.run(5, 4) # 采样点数为 5
print('区域内部的采样点坐标：', in_point, sep='\n')
