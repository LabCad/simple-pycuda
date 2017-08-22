#!/usr/bin/python
# basic example from SimplePyCuda library
# MIT License - Igor Machado Coelho (2017)

import ctypes

from simplepycuda import SimplePyCuda, SimpleSourceModule, grid, block

import numpy

def main():
	cuda = SimplePyCuda()
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
	d = numpy.random.randn(4,4)
	d = d.astype(numpy.float32)
	d_gpu = cuda.mem_alloc(d.nbytes)
	cuda.memcpy_htod(d_gpu, d)
	mod = SimpleSourceModule(""" __global__ void doublify ( float* d )
	  {
	    //int idx = threadIdx.x + threadIdx.y*4;
	    //d[idx] *= 2;
            //printf("oi=%d\\n",idx);
	  }
	""")
	func = mod.get_function("doublify")
	func(a_gpu, grid(1,2), block(3,4,5), 6, 7)
	cuda.deviceSynchronize()

	#
	print "will reset device"
	cuda.deviceReset()
	print '============ Finish SimplePyCuda example ============'

if __name__ == '__main__':
	main()


