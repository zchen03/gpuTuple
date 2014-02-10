#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define BLOCKS   3
#define DATASIZE BLOCKS * 512
#define CHUNKNUM 1
#define N		 100


cudaError_t square(int *result, int *data, int chunknum);

__global__ void squareKernel(int *result, int *data) {
	int i = threadIdx.x;
	result[i] = 0;
	result[i] = data[i] * data[i];
}

void deviceReset() {
// cudaDeviceReset must be called before exiting in order for profiling and
// tracing tools such as Parallel Nsight and Visual Profiler to show complete traces.
	cudaError_t cudaStatus;
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		//return 1;
	}

}

void setInput(int *data) {
	// Generate input data
	for (int i = 0; i < DATASIZE; i++) {
		data[i] = i;
	}
}

void printArray(char *content, int *input) {
	printf("%s\n", content);
	// Print the result array
	for (int i = 0; i < DATASIZE; i++)
		printf("i%d=%d, ", i, input[i]);
	printf("\n");
}

void dummyCudaMalloc(int **dummy_ptr) {
	cudaError_t cudaStatus = cudaMalloc((void**) dummy_ptr, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}
}

int main() {
	int data[DATASIZE];
	int result[DATASIZE] = { 0 };
	int *dummy_ptr = 0;
	dummyCudaMalloc(&dummy_ptr);
	// Set false value in result array
	memset(result, 0, DATASIZE);
	setInput(data);

	// Print the input character
	// printArray("Input", data);
	// Search keyword in parallel.
	printf("square\n");

	cudaError_t cudaStatus = square(result, data, CHUNKNUM);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}

	// printArray("Result", result);

	cudaFree(dummy_ptr);
	deviceReset();
	return 0;
}

// Helper function for using CUDA to search a list of characters in parallel.
cudaError_t square(int *result, int *data, int num_kernel) {
	int *dev_data = 0;
	int *dev_result = 0;
	cudaError_t cudaStatus;
	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr,
				"cudaSetDevice failed! Do you have a CUDA-capable GPU installed?");
		cudaFree(dev_result);
		return cudaStatus;
	}

	clock_t t = clock();
// Launch a search keyword kernel on the GPU with one thread for each element.
	for (int n = 0; n < N; n++) {
		for (int i = 0; i < num_kernel; i++) {
			int chunk_size = DATASIZE / num_kernel;
			// Allocate GPU buffers for result set.
			cudaStatus = cudaMalloc((void**) &dev_result, chunk_size * sizeof(int));
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMalloc failed!");
				goto Error;
			}
			// Allocate GPU buffers for data set.
			cudaStatus = cudaMalloc((void**) &dev_data, chunk_size * sizeof(int));
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMalloc failed!");
				goto Error;
			}
			// Copy input data from host memory to GPU buffers.
			cudaStatus = cudaMemcpy(dev_data, data + i * chunk_size,
					chunk_size * sizeof(int), cudaMemcpyHostToDevice);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMemcpy failed!");
				goto Error;
			}

			squareKernel<<<BLOCKS, chunk_size>>>(dev_result, dev_data);

			// cudaDeviceSynchronize waits for the kernel to finish, and returns
			// any errors encountered during the launch.
			cudaStatus = cudaDeviceSynchronize();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr,
						"cudaDeviceSynchronize returned error code %d after launching addKernel!\n",
						cudaStatus);
				goto Error;
			}
			// Copy result from GPU buffer to host memory.
			cudaStatus = cudaMemcpy(result + i * chunk_size, dev_result,
					chunk_size * sizeof(int), cudaMemcpyDeviceToHost);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMemcpy failed!");
				goto Error;
			}

			cudaFree(dev_data);
			cudaFree(dev_result);
			dev_data = NULL;
			dev_result = NULL;
		}
	}

	t = clock() - t;
	printf("%d kernel time for %d kernel(s): %f milliseconds\n",
			N, CHUNKNUM,
			((float) t) * 1000.0 / CLOCKS_PER_SEC);
	Error: cudaFree(dev_result);
	return cudaStatus;

}
