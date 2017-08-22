all:
	nvcc --shared cudapp.cu -o cudapp.so --compiler-options -fPIC

run:
	export LD_LIBRARY_PATH=. && ./testSimplePyCuda.py

clean:
	rm *.so
