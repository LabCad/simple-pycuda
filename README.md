The library cudapp is the basis of SimplePyCuda, a simple wrapper for CUDA functions in Python.
The idea is to workaround the issues regarding Context in old CUDA versions, that persist on PyCuda.

No need to pre-compile the project.

Install from pip:
```bash
pip install simplepycuda
```



Simple code example.
==========================

In Python:

```python
import ctypes
import numpy
from simplepycuda import SimplePyCuda, SimpleSourceModule, grid, block

def classicExample(cuda):
	a = numpy.random.randn(4,4)
	a = a.astype(numpy.float32)
	print a
	a_gpu = cuda.mem_alloc(a.nbytes)
	cuda.memcpy_htod(a_gpu, a)
	mod = SimpleSourceModule("""
          #include<stdio.h>
          __global__ void doublify ( float* a )
	  {
	    int idx = threadIdx.x + threadIdx.y*4;
	    a[idx] *= 2;
            //printf("oi=%d\\n",idx);
	  }
	""")
	func = mod.get_function("doublify")
	# TODO: this next line will be made automatically in get_function method... just need a few more time :)
	func.argtypes = [ctypes.c_void_p, grid, block, ctypes.c_ulong, ctypes.c_ulong]
	func(a_gpu, grid(1,1), block(4,4,1), 0, 0)
	cuda.memcpy_dtoh(a, a_gpu)
	cuda.deviceSynchronize()
	print a
	cuda.free(a_gpu) # this is not necessary in PyCUDA
	print "Finished"

def main():
	cuda = SimplePyCuda()

	classicExample(cuda)	
	return 0
```

[MIT License](LICENSE) - Igor Machado Coelho and Rodolfo Pereira Araujo (2017)

