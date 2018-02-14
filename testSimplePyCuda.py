#!/usr/bin/python
# basic example from SimplePyCuda library
# MIT License - Igor Machado Coelho (2017)

import ctypes

from simplepycuda import SimplePyCuda, SimpleSourceModule, Grid, Block

import numpy

def simpleLoadTest(cuda):
	lib = ctypes.cdll.LoadLibrary('./__simplepycuda_kernel_doublify.so')
	lib.kernel_loader.argtypes = [ctypes.c_void_p, Grid, Block, ctypes.c_ulong, ctypes.c_ulong]
	a = numpy.random.randn(4,4)
	a_gpu = cuda.mem_alloc(a.nbytes)
	lib.kernel_loader(a_gpu, Grid(1, 1), Block(4, 4, 1), 0, 0)
	print "Kernel OK"
	# finish

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
            printf("oi=%d\\n",idx);
	  }
	""","nvcc", ["--ptxas-options=-v","--compiler-options -O3","--compiler-options -Wall"])
	func = mod.get_function("doublify")
	# TODO: this next line will be made automatically in get_function method... just need a few more time :)
	func.argtypes = [ctypes.c_void_p, Grid, Block, ctypes.c_ulong, ctypes.c_ulong]
	func(a_gpu, Grid(1, 1), Block(4, 4, 1), 0, 0)
	cuda.memcpy_dtoh(a, a_gpu)
	cuda.deviceSynchronize()
	print a
	cuda.free(a_gpu) # this is not necessary in PyCUDA
	print "Finished"

def main():
	cuda = SimplePyCuda()

	classicExample(cuda)	
	return 0

	print '============ SimplePyCuda ============'
	p = cuda.malloc(10)
	print "malloc'd 10 bytes. GPU pointer:", p
	cuda.free(p)
	print "free'd 10 bytes at", p
	#
	print "will reset device"
	cuda.deviceReset()
	#
	print "number of GPU devices:",cuda.getDeviceCount()
	print "selecting GPU device 0"
	cuda.setDevice(0)
	#
	print "creating float matrix"
	a = numpy.random.randn(4,3)
	print(a)
	a = a.astype(numpy.float32)
	print a.nbytes, "bytes"
	a_gpu = cuda.mem_alloc(a.nbytes)
	print "GPU pointer=",a_gpu
	cuda.memcpy_htod(a_gpu, a)
	cuda.memcpy_dtoh(a, a_gpu)
	print(a)
	#
	cuda.free(a_gpu)
	#
	print "creating integer matrix (as vector)"
	b = numpy.random.randn(4,3)*10
	b = b.astype(numpy.int32)
	print(b)
	b_gpu = cuda.mem_alloc(b.nbytes)
	cuda.memcpy_htod(b_gpu, b)
	#
	print "filling first 4 bytes (first integer) with 0x00, 0 in decimal"
	cuda.memset(b_gpu, 0x00, 4)
	cuda.memcpy_dtoh(b, b_gpu)
	print(b)
	#
	print "filling first 10 bytes (first two integers and half) with 0x12, 303174162 in decimal"
	cuda.memset(b_gpu, 0x12, 10)
	cuda.memcpy_dtoh(b, b_gpu)
	print(b)
	#
	print "filling all bytes with 0x12 (303174162 in decimal)"
	cuda.memset(b_gpu, 0x12, b.nbytes)
	cuda.memcpy_dtoh(b, b_gpu)
	print(b)
	#
	print "synchronizing device"
	cuda.deviceSynchronize()
	#
	c = numpy.random.randn(1000,1000)
	c = c.astype(numpy.float32)
	t1 = cuda.eventCreate()
	t2 = cuda.eventCreate()
	cuda.eventRecord(t1)
	c_gpu = cuda.mem_alloc(c.nbytes)
	cuda.memcpy_htod(c_gpu, c)
	cuda.memcpy_dtoh(c, c_gpu)
	cuda.eventRecord(t2)
	cuda.eventSynchronize(t2) # if not used, will issue a 'cudaErrorNotReady' error.
	time = cuda.eventElapsedTime(t1, t2)
	print "took",time,"ms to copy",c.nbytes,"bytes"
	cuda.eventDestroy(t1)
	cuda.eventDestroy(t2)
	#
	print ""
	print "will test doublify kernel"
	cuda.memcpy_htod(a_gpu, a)
	print a
	mod = SimpleSourceModule(""" __global__ void doublify ( float* a )
	  {
	    int idx = threadIdx.x + threadIdx.y*4;
	    a[idx] *= 2;
            //printf("oi=%d\\n",idx);
	  }
	""")
	func = mod.get_function("doublify")
	# TODO: this will enter automatically in get_function method... just need a few more time :)
	func.argtypes = [ctypes.c_void_p, Grid, Block, ctypes.c_ulong, ctypes.c_ulong]
	func(a_gpu, Grid(1, 1), Block(4, 4, 1), 0, 0)
	cuda.deviceSynchronize()
	print "kernel executed"
	cuda.memcpy_dtoh(a, a_gpu)
	print a
	#
	print "will reset device"
	cuda.deviceReset()
	print '============ Finish SimplePyCuda example ============'

if __name__ == '__main__':
	main()


