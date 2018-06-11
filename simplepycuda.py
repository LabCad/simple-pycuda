#!/usr/bin/python

import ctypes
from ctypes import *   #import Structure

import os
import os.path
import re


class SimplePyCuda:
	def __init__(self, path="./"):
		cudaappfile = 'cudapp'
		if not os.path.isfile(path + cudaappfile + '.so'):
			SimpleSourceModule.compile_files('nvcc', [path + cudaappfile + '.cu'], [])

		self.lib = ctypes.cdll.LoadLibrary(path + cudaappfile + '.so')
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
		return self.lib.cudappEventCreate()

	def eventDestroy(self, event):
		self.lib.cudappEventDestroy(event)

	def eventSynchronize(self, event):
		self.lib.cudappEventSynchronize(event)

	def eventRecord(self, event, stream=0):
		self.lib.cudappEventRecord(event, stream)

	def eventElapsedTime(self, event1, event2):
		return self.lib.cudappEventElapsedTime(event1, event2)


class Grid(Structure):
	_fields_ = [("x", c_int), ("y", c_int)]


class Block(Structure):
	_fields_ = [("x", c_int), ("y", c_int), ("z", c_int)]


class FunctionToCall:
	def __init__(self, func, func_params, _grid=(1, 1), _block=(4, 4, 1), sharedsize=0, streamid=0):
		self.__func = func
		self.__grid = Grid(_grid[0], _grid[1])
		self.__block = Block(_block[0], _block[1], _block[2])
		self.__sharedsize = sharedsize
		self.__streamid = streamid
		self.set_arguments(self.__get_param_type(func_params))

	def __call__(self, *args, **kwargs):
		return self.__func(*(args + (self.__grid, self.__block, self.__sharedsize, self.__streamid)), **kwargs)

	@property
	def sharedsize(self):
		return self.__sharedsize

	@property
	def streamid(self):
		return self.__streamid

	@property
	def block(self):
		return self.__block

	@property
	def grid(self):
		return self.__grid

	@property
	def argtypes(self):
		return self.__func.argtypes

	def __get_param_type(self, func_params):
		rem_spc = re.compile('\s+')
		resp = []
		name_param =\
			{
				'bool': ctypes.c_bool,
				'char': ctypes.c_char,
				'short': ctypes.c_short,
				'unsigned short': ctypes.c_ushort,
				'int': ctypes.c_int,
				'unsigned int': ctypes.c_uint,
				'long': ctypes.c_long,
				'unsigned long': ctypes.c_ulong,
				'long long': ctypes.c_longlong,
				'unsigned long long': ctypes.c_ulonglong,
				'float': ctypes.c_float,
				'double': ctypes.c_double,
				'long double': ctypes.c_longdouble
			}
		for i in xrange(0, len(func_params), 2):
			param_type = rem_spc.sub(' ', func_params[i])
			if param_type in name_param:
				resp.append(name_param[param_type])
			elif '*' in param_type:
				resp.append(ctypes.c_void_p)
			else:
				raise ValueError("Type '{}' not found, of list {}".format(param_type, func_params))
		return resp

	def set_arguments(self, arglist):
		if self.__func.argtypes is None or len(self.__func.argtypes) != 4:
			self.__func.argtypes = arglist + [Grid, Block, ctypes.c_ulong, ctypes.c_ulong]
		elif len(self.__func.argtypes) == 4:
			self.__func.argtypes = arglist + self.__func.argtypes


class SimpleSourceModule:
	def __init__(self, kernelcode, nvcc='nvcc', options=[]):
		self.code = kernelcode
		self.nvcc = nvcc
		self.options = options

	@staticmethod
	def __gen_cufile(function_name, before, klist, splitcode, nvcc, options):
		with open("__simplepycuda_kernel_" + function_name + ".cu", "w") as f:
			f.write(before)
			f.write("\n\n")
			f.write("struct simplepycuda_grid { int x,y; };\n\n")
			f.write("struct simplepycuda_block { int x,y,z; };\n\n")
			f.write("__global__ void kernel_")
			f.write(function_name)
			f.write("( ")

			# i = 4
			# while i < len(klist) - 1:
			for i in xrange(4, len(klist) - 1, 2):
				f.write(klist[i])  # variable type
				f.write(" ")
				f.write(klist[i + 1])  # variable name
				if i + 2 < len(klist) - 1:
					f.write(" , ")

			f.write(" )\n")
			f.write(splitcode[1])
			f.write("\nextern \"C\" void ")
			f.write("kernel_loader")
			# f.write(function_name)
			f.write("( ")

			# while i < len(klist) - 1:
			for i in xrange(4, len(klist) - 1, 2):
				f.write(klist[i])  # variable type
				f.write(" ")
				f.write(klist[i + 1])  # variable name
				if i + 2 < len(klist) - 1:
					f.write(" , ")

			f.write(" , simplepycuda_grid g, simplepycuda_block b, size_t shared, size_t stream) {\n")
			# f.write("//\tprintf(\"lets go! grid(%d,%d) block(%d,%d,%d) shared=%lu stream=%lu\\n\",g.x,g.y,b.x,b.y,b.z,shared,stream);\n")
			f.write("\tdim3 mygrid;  mygrid.x = g.x;  mygrid.y = g.y;\n")
			f.write("\tdim3 myblock; myblock.x = b.x; myblock.y = b.y; myblock.z = b.z;\n")
			f.write("\tkernel_")
			f.write(function_name)
			f.write("<<<mygrid, myblock, shared, cudaStream_t(stream)>>>( ")
			# while i < len(klist) - 1:
			for i in xrange(4, len(klist) - 1, 2):
				f.write(klist[i + 1])  # variable name
				if i + 2 < len(klist) - 1:
					f.write(" , ")

			f.write(");\n")
			f.write("cudaDeviceSynchronize();\n");
			# f.write("//\tprintf(\"finished kernel!\");\n")
			f.write("}\n")
			f.write("\n\n//")
			compilecommand = SimpleSourceModule.__get_compile_command(nvcc,
				SimpleSourceModule.__get_file_name(function_name) + ".cu", options)
			f.write(compilecommand)
			f.write("\n")

	@staticmethod
	def __get_os_function(loadkernelpath, func_params):
		kernelfunction = ctypes.cdll.LoadLibrary(loadkernelpath)
		# TODO: add argtypes here in function kernel_loader!
		# kernelfunction.kernel_loader.argtypes = [ctypes.c_void_p, grid, block, ctypes.c_ulong, ctypes.c_ulong]
		return FunctionToCall(kernelfunction.kernel_loader, func_params)

	@staticmethod
	def __get_file_name(function_name):
		return "__simplepycuda_kernel_" + function_name

	@staticmethod
	def __get_compile_command(nvcc, filename, options, objectname=None, compiler_options=[]):
		objectname = objectname or filename
		if objectname.lower().endswith(".cu"):
			objectname = objectname[:-3]
		if objectname.lower().endswith(".cpp"):
			objectname = objectname[:-4]

		compilecommand = "{} --shared {}".format(nvcc, filename)
		compilecommand = "{} {}".format(compilecommand, " ".join(options))
		return "{} -o {}.so --compiler-options -fPIC {} 2> {}.log".format(compilecommand, objectname,
			" ".join(compiler_options), objectname)

	@staticmethod
	def compile_files(nvcc, files, options, objectname=None, compiler_options=[]):
		objectname = objectname or files[0]
		compilecommand = SimpleSourceModule.__get_compile_command(nvcc, " ".join(files), options, objectname, compiler_options)
		oscode = os.system(compilecommand)
		if oscode != 0:
			print "ERROR: compile error for kernel! view log file for more information!"
			assert False

	def get_function(self, function_name_input, cache_function=True):
		if re.match("[_A-Za-z][_a-zA-Z0-9]*$", function_name_input) is None:
			print "ERROR: kernel name is not valid '", function_name_input, "'"
			assert False
		function_name = re.match("[_A-Za-z][_a-zA-Z0-9]*$", function_name_input).group(0)
		globalword = '__global__'
		id_global = self.code.find(globalword)
		before = self.code[:id_global]
		after = self.code[id_global:]
		splitcode = after.split('\n', 1)
		kernel_signature = splitcode[0]
		fn = re.compile('.*(__global__)\s*(void)\s*(\S*)\s*\(')
		ftok = fn.match(kernel_signature)
		klist = [ftok.group(1), ftok.group(2), ftok.group(3), '(', ')']
		params_rexp = re.compile('.*\((.*)\)')
		rem_ast = re.compile('\s*\*\s*')
		# Removendo espaco antes do *
		func_param_str = rem_ast.sub('* ', params_rexp.match(kernel_signature).group(1)).split(',')
		func_params = [item for sublist in [param.split() for param in func_param_str] for item in sublist]
		klist = klist[:4] + func_params + [klist[-1]]
		assert klist[0] == globalword
		assert klist[1] == "void"
		assert klist[2] == function_name
		assert klist[3] == "("
		assert klist[len(klist)-1] == ")"
		# cufile = "__simplepycuda_kernel_" + function_name + ".cu"
		loadkernelpath = "./" + SimpleSourceModule.__get_file_name(function_name) + ".so"
		if cache_function and os.path.isfile(loadkernelpath):
			return SimpleSourceModule.__get_os_function(loadkernelpath)

		SimpleSourceModule.__gen_cufile(function_name, before, klist, splitcode, self.nvcc, self.options)

		SimpleSourceModule.compile_files(self.nvcc,
			[SimpleSourceModule.__get_file_name(function_name) + ".cu"], self.options)

		return SimpleSourceModule.__get_os_function(loadkernelpath, func_params)

	def get_function_debug(self, function_name):
		print "Will debug kernel function call for '", function_name, "'! This is a development-only feature!"
		print self.code
		print "function_name =", function_name
		kernel_signature = self.code.split('\n', 1)[0]
		print "header: ", kernel_signature
		print "WARNING: spaces must be provided in order to properly tokenize!"

		klist = kernel_signature.split()
		print klist
		assert klist[0] == "__global__"
		assert klist[1] == "void"
		assert klist[2] == function_name
		assert klist[3] == "("
		assert klist[len(klist)-1] == ")"
		i = 4
		while i < len(klist)-1:
			print "variable type: ", klist[i]
			print "variable name: ", klist[i+1]
			if klist[i+1][0] == "*":
				print "ERROR: pointers should be put together with TYPES, not variables.. please follow C++ style :)"
				assert False
			if i+2 < len(klist)-1:
				assert klist[i+2] == ","
			i += 3
		print "Kernel parameters seem ok :)"
		return None
