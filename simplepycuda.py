#!/usr/bin/python

import ctypes

class SimplePyCuda:
	def __init__(self):
		self.lib = ctypes.cdll.LoadLibrary('./cudapp.so')
		self.lib.cudappMemcpyHostToDevice.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
		self.lib.cudappMemcpyDeviceToHost.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
		#
		self.lib.cudappMalloc.argtypes = [ctypes.c_ulong]
		self.lib.cudappMalloc.restype = ctypes.c_void_p
		self.lib.cudappFree.argtypes = [ctypes.c_void_p]
		self.lib.cudappMemset.argtypes = [ctypes.c_void_p, ctypes.c_byte, ctypes.c_ulong]
		#
		self.lib.cudappEventCreate.restype = ctypes.c_void_p
		self.lib.cudappEventDestroy.argtypes = [ctypes.c_void_p]
		self.lib.cudappEventSynchronize.argtypes = [ctypes.c_void_p]
		self.lib.cudappEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
		self.lib.cudappEventElapsedTime.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
		self.lib.cudappEventElapsedTime.restype = ctypes.c_float

	def getDeviceCount(self):
		return self.lib.cudappGetDeviceCount()

	def setDevice(self, index):
		self.lib.cudappSetDevice(index)

	def deviceSynchronize(self):
		self.lib.cudappDeviceSynchronize()

	def deviceReset(self):
		self.lib.cudappDeviceReset()

	def malloc(self, nbytes):
		return self.lib.cudappMalloc(nbytes)
	# pycuda compatible version
	def mem_alloc(self, nbytes): 	        
		return self.malloc(nbytes)

	def free(self, gpu_pointer):
		return self.lib.cudappFree(gpu_pointer)
	
	def memset(self, gpu_pointer, bytevalue, count):
		return self.lib.cudappMemset(gpu_pointer, bytevalue, count)

	def memcpy_htod(self, d, h, nbytes):
		return self.lib.cudappMemcpyHostToDevice(d,h,nbytes)
	# pycuda compatible version
	def memcpy_htod(self, gpu_pointer, cpu):
		self.lib.cudappMemcpyHostToDevice(gpu_pointer, ctypes.c_void_p(cpu.ctypes.data), cpu.nbytes)		

	def memcpy_dtoh(self, h, d, nbytes):
		return self.lib.cudappMemcpyDeviceToHost(h,d,nbytes)
	# pycuda compatible version
	def memcpy_dtoh(self, cpu, gpu_pointer):
		self.lib.cudappMemcpyDeviceToHost(ctypes.c_void_p(cpu.ctypes.data), gpu_pointer, cpu.nbytes)		


	def eventCreate(self):
		return	self.lib.cudappEventCreate()
	def eventDestroy(self, event):
		self.lib.cudappEventDestroy(event)
	def eventSynchronize(self, event):
		self.lib.cudappEventSynchronize(event)
	def eventRecord(self, event, stream=0):
		self.lib.cudappEventRecord(event, stream)
	def eventElapsedTime(self, event1, event2):
		return self.lib.cudappEventElapsedTime(event1, event2)


