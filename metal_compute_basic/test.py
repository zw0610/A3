import objcpt
import numpy as np

ot = objcpt.objcpp_test()
ot.print_device()

a = np.random.rand(5).astype(np.float32)
b = np.random.rand(5).astype(np.float32)
r = np.zeros_like(a)

ot.run_metal(a, b, r)

print(r)
print(a+b)