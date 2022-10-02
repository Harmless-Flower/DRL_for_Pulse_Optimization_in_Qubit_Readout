import numpy as np

length = 50
start = 0.0
end = 2.0
actual_start = (end - start)/length + start
print(actual_start)

t_list = np.linspace(actual_start, end, length)
print(t_list)

t_prim = t_list[0:25]
print(t_prim)
print(len(t_prim))

t_seg = t_list[25:50]
print(t_seg)
print(len(t_seg))
