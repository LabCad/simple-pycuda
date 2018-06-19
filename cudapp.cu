#include "cudapp.h"

extern "C" int cudappGetDeviceCount() {
	int deviceCount;
	checkCudaErrors(cudaGetDeviceCount(&deviceCount));
	return deviceCount;
}

extern "C" void cudappSetDevice(int index) {
	checkCudaErrors(cudaSetDevice(index));
}

extern "C" void cudappDeviceSynchronize() {
	checkCudaErrors(cudaDeviceSynchronize());
}

extern "C" void cudappDeviceReset() {
	checkCudaErrors(cudaDeviceReset());
}

extern "C" void* cudappMalloc(size_t nbytes) {
	void* p;
	checkCudaErrors(cudaMalloc(&p, nbytes));
	return p;
}

extern "C" void cudappFree(void* p) {
	checkCudaErrors(cudaFree(p));
}

extern "C" void cudappMemset(void* p, unsigned char v, size_t count) {
	checkCudaErrors(cudaMemset(p, v, count));
}

extern "C" void cudappMemcpyHostToDevice(void* d, void* h, size_t nbytes) {
	checkCudaErrors(cudaMemcpy(d, h, nbytes, cudaMemcpyHostToDevice));
}

extern "C" void cudappMemcpyDeviceToHost(void* h, void* d, size_t nbytes) {
	checkCudaErrors(cudaMemcpy(h, d, nbytes, cudaMemcpyDeviceToHost));
}

// event handling (timer)

extern "C" void* cudappEventCreate() {
	cudaEvent_t* ev = new cudaEvent_t;
	checkCudaErrors(cudaEventCreate(ev));
	return ev;
}

extern "C" void cudappEventRecord(void* event, size_t stream) {
	cudaEvent_t* ev = (cudaEvent_t*) event;
	checkCudaErrors(cudaEventRecord(*ev, (cudaStream_t) stream));
}

extern "C" void cudappEventSynchronize(void* event) {
	cudaEvent_t* ev = (cudaEvent_t*) event;
	checkCudaErrors(cudaEventSynchronize(*ev));
}

extern "C" float cudappEventElapsedTime(void* event1, void* event2) {
	cudaEvent_t* ev1 = (cudaEvent_t*) event1;
	cudaEvent_t* ev2 = (cudaEvent_t*) event2;
	float f;
	checkCudaErrors(cudaEventElapsedTime(&f, *ev1, *ev2));
	return f;
}

extern "C" void cudappEventDestroy(void* event) {
	cudaEvent_t* ev = (cudaEvent_t*) event;
	checkCudaErrors(cudaEventDestroy(*ev));
	delete ev;
}

extern "C" cudaError_t cudappGetLastError() {
	return cudaGetLastError();
}
