#include <opencv2/opencv.hpp>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdint.h>

#define ck(call) \
	do { \
		cudaError_t err__ = call; \
		if (err__ != cudaSuccess) { \
			OnCudaError(err__); \
				} \
		} while (0)

void OnCudaError(cudaError_t err) {
	printf("[CUDA Error] %d: %s", err, cudaGetErrorString(err));
	throw "CUDA ERROR";
}

void GaussianKernelCPU(const uint8_t *src, uint8_t *dst, int width, int height, int step)
{
	const float filter[5][5] = {
		{ 0.002969017f, 0.01330621f, 0.021938231f, 0.01330621f, 0.002969017f },
		{ 0.01330621f, 0.059634295f, 0.098320331f, 0.059634295f, 0.01330621f },
		{ 0.021938231f, 0.098320331f, 0.162102822f, 0.098320331f, 0.021938231f },
		{ 0.01330621f, 0.059634295f, 0.098320331f, 0.059634295f, 0.01330621f },
		{ 0.002969017f, 0.01330621f, 0.021938231f, 0.01330621f, 0.002969017f },
	};
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			float sum = 0;
			for (int dy = 0; dy < 5; ++dy) {
				for (int dx = 0; dx < 5; ++dx) {
					sum += filter[dy][dx] * src[(x + dx) + (y + dy) * step];
				}
			}
			dst[x + y * step] = (int)(sum + 0.5f);
		}
	}
}

__global__ void GaussianKernelSimple(const uint8_t *src, uint8_t *dst, int width, int height, int step)
{
	const float filter[5][5] = {
		{ 0.002969017f, 0.01330621f, 0.021938231f, 0.01330621f, 0.002969017f },
		{ 0.01330621f, 0.059634295f, 0.098320331f, 0.059634295f, 0.01330621f },
		{ 0.021938231f, 0.098320331f, 0.162102822f, 0.098320331f, 0.021938231f },
		{ 0.01330621f, 0.059634295f, 0.098320331f, 0.059634295f, 0.01330621f },
		{ 0.002969017f, 0.01330621f, 0.021938231f, 0.01330621f, 0.002969017f },
	};

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height) {
		float sum = 0;
		for (int dy = 0; dy < 5; ++dy) {
			for (int dx = 0; dx < 5; ++dx) {
				sum += filter[dy][dx] * src[(x + dx) + (y + dy) * step];
			}
		}
		dst[x + y * step] = (int)(sum + 0.5f);
	}
}

__global__ void GaussianKernelArray(const uint8_t *src, uint8_t *dst, int width, int height, int step, int ks)
{
	const float filter[5][5] = {
		{ 0.002969017f, 0.01330621f, 0.021938231f, 0.01330621f, 0.002969017f },
		{ 0.01330621f, 0.059634295f, 0.098320331f, 0.059634295f, 0.01330621f },
		{ 0.021938231f, 0.098320331f, 0.162102822f, 0.098320331f, 0.021938231f },
		{ 0.01330621f, 0.059634295f, 0.098320331f, 0.059634295f, 0.01330621f },
		{ 0.002969017f, 0.01330621f, 0.021938231f, 0.01330621f, 0.002969017f },
	};

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height) {
		float sum = 0;
		for (int dy = 0; dy < ks; ++dy) {
			for (int dx = 0; dx < ks; ++dx) {
				sum += filter[dy][dx] * src[(x + dx) + (y + dy) * step];
			}
		}
		dst[x + y * step] = (int)(sum + 0.5f);
	}
}

__constant__ float filter[5][5] = {
	{ 0.002969017f, 0.01330621f, 0.021938231f, 0.01330621f, 0.002969017f },
	{ 0.01330621f, 0.059634295f, 0.098320331f, 0.059634295f, 0.01330621f },
	{ 0.021938231f, 0.098320331f, 0.162102822f, 0.098320331f, 0.021938231f },
	{ 0.01330621f, 0.059634295f, 0.098320331f, 0.059634295f, 0.01330621f },
	{ 0.002969017f, 0.01330621f, 0.021938231f, 0.01330621f, 0.002969017f },
};

__global__ void GaussianKernelConstant(const uint8_t *src, uint8_t *dst, int width, int height, int step, int ks)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height) {
		float sum = 0;
		for (int dy = 0; dy < ks; ++dy) {
			for (int dx = 0; dx < ks; ++dx) {
				sum += filter[dy][dx] * src[(x + dx) + (y + dy) * step];
			}
		}
		dst[x + y * step] = (int)(sum + 0.5f);
	}
}

__global__ void GaussianKernelConstantFixed(const uint8_t *src, uint8_t *dst, int width, int height, int step)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height) {
		float sum = 0;
		for (int dy = 0; dy < 5; ++dy) {
			for (int dx = 0; dx < 5; ++dx) {
				sum += filter[dy][dx] * src[(x + dx) + (y + dy) * step];
			}
		}
		dst[x + y * step] = (int)(sum + 0.5f);
	}
}

__global__ void GaussianKernelShared(const uint8_t *src, uint8_t *dst, int width, int height, int step)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	__shared__ uint8_t sbuf[8+4][32+4];

	if (x < width && y < height) {
		sbuf[ty][tx] = src[x + y * step];
		if (tx < 4 && x + 32 < width + 4) {
			sbuf[ty][tx + 32] = src[x + 32 + y * step];
		}
		if (ty < 4 && y + 8 < height + 4) {
			sbuf[ty + 8][tx] = src[x + (y + 8) * step];
			if (tx < 4 && x + 32 < width + 4) {
				sbuf[ty + 8][tx + 32] = src[x + 32 + (y + 8) * step];
			}
		}
	}

	__syncthreads();

	if (x < width && y < height) {
		float sum = 0;
		for (int dy = 0; dy < 5; ++dy) {
			for (int dx = 0; dx < 5; ++dx) {
				sum += filter[dy][dx] * sbuf[ty + dy][tx + dx];
			}
		}
		dst[x + y * step] = (int)(sum + 0.5f);
	}
}

cv::Mat GaussianFilterGPUSimple(cv::Mat src)
{
	int width = src.cols, height = src.rows;
	uint8_t *dev_src, *dev_dst;

	ck(cudaMalloc((void**)&dev_src, width * height * sizeof(uint8_t)));
	ck(cudaMalloc((void**)&dev_dst, width * height * sizeof(uint8_t)));
	ck(cudaMemcpy(dev_src, src.data, width * height * sizeof(uint8_t), cudaMemcpyHostToDevice));

    // Launch a kernel on the GPU with one thread for each element.
	dim3 threadsPerBlock(32, 32);
	dim3 numBlocks(
		(width + threadsPerBlock.x - 1) / threadsPerBlock.x,
		(height + threadsPerBlock.y - 1) / threadsPerBlock.y);
	GaussianKernelSimple <<<numBlocks, threadsPerBlock >>>(dev_src, dev_dst, width - 4, height - 4, width);

    // Check for any errors launching the kernel
	ck(cudaGetLastError());
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
	ck(cudaDeviceSynchronize());

	cv::Mat dst(src.rows, src.cols, src.type());

    // Copy output vector from GPU buffer to host memory.
	ck(cudaMemcpy(dst.data, dev_dst, width * height * sizeof(uint8_t), cudaMemcpyDeviceToHost));

	ck(cudaFree(dev_src));
	ck(cudaFree(dev_dst));

	return dst;
}

cv::Mat GaussianFilterGPUOpt(cv::Mat src, int opt)
{
	int width = src.cols, height = src.rows;
	uint8_t *dev_src, *dev_dst;

	ck(cudaMalloc((void**)&dev_src, width * height * sizeof(uint8_t)));
	ck(cudaMalloc((void**)&dev_dst, width * height * sizeof(uint8_t)));
	ck(cudaMemcpy(dev_src, src.data, width * height * sizeof(uint8_t), cudaMemcpyHostToDevice));

	// Launch a kernel on the GPU with one thread for each element.
	dim3 threadsPerBlock(32, 8);
	dim3 numBlocks(
		(width + threadsPerBlock.x - 1) / threadsPerBlock.x,
		(height + threadsPerBlock.y - 1) / threadsPerBlock.y);
	if (opt == 1)
		GaussianKernelArray << <numBlocks, threadsPerBlock >> > (dev_src, dev_dst, width - 4, height - 4, width, 5);
	else if (opt == 2)
		GaussianKernelConstant << <numBlocks, threadsPerBlock >> > (dev_src, dev_dst, width - 4, height - 4, width, 5);
	else if (opt == 3)
		GaussianKernelConstantFixed << <numBlocks, threadsPerBlock >> > (dev_src, dev_dst, width - 4, height - 4, width);
	else if (opt == 4)
		GaussianKernelShared << <numBlocks, threadsPerBlock >> > (dev_src, dev_dst, width - 4, height - 4, width);
	else
		printf("NOT IMPLEMENTED\n");

	// Check for any errors launching the kernel
	ck(cudaGetLastError());

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	ck(cudaDeviceSynchronize());

	cv::Mat dst(src.rows, src.cols, src.type());

	// Copy output vector from GPU buffer to host memory.
	ck(cudaMemcpy(dst.data, dev_dst, width * height * sizeof(uint8_t), cudaMemcpyDeviceToHost));

	ck(cudaFree(dev_src));
	ck(cudaFree(dev_dst));

	return dst;
}

__constant__ float filter3[3][3] = {
	{ 0.059634295f, 0.098320331f, 0.059634295f },
	{ 0.098320331f, 0.162102822f, 0.098320331f },
	{ 0.059634295f, 0.098320331f, 0.059634295f },
};

__global__ void BilateralKernelNaive(const uint8_t *src, uint8_t *dst, int width, int height, int step, float sigma)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height) {
		float c_sum = 0;
		float f_sum = 0;
		int val0 = src[x + y * step];
		for (int dy = 0; dy < 3; ++dy) {
			for (int dx = 0; dx < 3; ++dx) {
				int val = src[(x + dx) + (y + dy) * step];
				int diff = val - val0;
				float w = filter3[dy][dx] * (1 / sqrtf(2 * 3.1415926f * sigma * sigma)) * expf(-diff * diff / (2 * sigma * sigma));
				f_sum += w;
				c_sum += w * val;
			}
		}
		dst[x + y * step] = (int)(c_sum / f_sum + 0.5f);
	}
}

__global__ void BilateralKernelSimple(const uint8_t *src, uint8_t *dst, int width, int height, int step, float sigma)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height) {
		float coef = 1.0 / sqrtf(2 * 3.1415926f * sigma * sigma);
		float coef2 = -1.0 / (2 * sigma * sigma);
		float c_sum = 0;
		float f_sum = 0;
		int val0 = src[x + y * step];
		for (int dy = 0; dy < 3; ++dy) {
			for (int dx = 0; dx < 3; ++dx) {
				int val = src[(x + dx) + (y + dy) * step];
				int diff = val - val0;
				float w = filter3[dy][dx] * coef * expf(diff * diff * coef2);
				f_sum += w;
				c_sum += w * val;
			}
		}
		dst[x + y * step] = (int)(c_sum / f_sum + 0.5f);
	}
}

__global__ void BilateralKernelFast(const uint8_t *src, uint8_t *dst, int width, int height, int step, float sigma)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height) {
		float coef = __frsqrt_rn(2 * 3.1415926f * sigma * sigma);
		float coef2 = __frcp_rn(-2 * sigma * sigma);
		float c_sum = 0;
		float f_sum = 0;
		int val0 = src[x + y * step];
		for (int dy = 0; dy < 3; ++dy) {
			for (int dx = 0; dx < 3; ++dx) {
				int val = src[(x + dx) + (y + dy) * step];
				int diff = val - val0;
				float w = filter3[dy][dx] * coef * __expf(diff * diff * coef2);
				f_sum += w;
				c_sum += w * val;
			}
		}
		dst[x + y * step] = (int)(__fdividef(c_sum, f_sum) + 0.5f);
	}
}

cv::Mat BilateralFilterGPUOpt(cv::Mat src, int opt)
{
	int width = src.cols, height = src.rows;
	uint8_t *dev_src, *dev_dst;

	ck(cudaMalloc((void**)&dev_src, width * height * sizeof(uint8_t)));
	ck(cudaMalloc((void**)&dev_dst, width * height * sizeof(uint8_t)));
	ck(cudaMemcpy(dev_src, src.data, width * height * sizeof(uint8_t), cudaMemcpyHostToDevice));

	// Launch a kernel on the GPU with one thread for each element.
	dim3 threadsPerBlock(32, 8);
	dim3 numBlocks(
		(width + threadsPerBlock.x - 1) / threadsPerBlock.x,
		(height + threadsPerBlock.y - 1) / threadsPerBlock.y);
	if (opt == 1)
		BilateralKernelNaive << <numBlocks, threadsPerBlock >> > (dev_src, dev_dst, width - 2, height - 2, width, 10);
	else if (opt == 2)
		BilateralKernelSimple << <numBlocks, threadsPerBlock >> > (dev_src, dev_dst, width - 2, height - 2, width, 10);
	else if (opt == 3)
		BilateralKernelFast << <numBlocks, threadsPerBlock >> > (dev_src, dev_dst, width - 2, height - 2, width, 10);
	else
		printf("NOT IMPLEMENTED\n");

	// Check for any errors launching the kernel
	ck(cudaGetLastError());

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	ck(cudaDeviceSynchronize());

	cv::Mat dst(src.rows, src.cols, src.type());

	// Copy output vector from GPU buffer to host memory.
	ck(cudaMemcpy(dst.data, dev_dst, width * height * sizeof(uint8_t), cudaMemcpyDeviceToHost));

	ck(cudaFree(dev_src));
	ck(cudaFree(dev_dst));

	return dst;
}

cv::Mat GaussianFilterGPUArray(cv::Mat src)
{
	int width = src.cols, height = src.rows;
	uint8_t *dev_src, *dev_dst;

	ck(cudaMalloc((void**)&dev_src, width * height * sizeof(uint8_t)));
	ck(cudaMalloc((void**)&dev_dst, width * height * sizeof(uint8_t)));
	ck(cudaMemcpy(dev_src, src.data, width * height * sizeof(uint8_t), cudaMemcpyHostToDevice));

	// Launch a kernel on the GPU with one thread for each element.
	dim3 threadsPerBlock(32, 8);
	dim3 numBlocks(
		(width + threadsPerBlock.x - 1) / threadsPerBlock.x,
		(height + threadsPerBlock.y - 1) / threadsPerBlock.y);
	GaussianKernelArray << <numBlocks, threadsPerBlock >> >(dev_src, dev_dst, width - 4, height - 4, width, 5);

	// Check for any errors launching the kernel
	ck(cudaGetLastError());

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	ck(cudaDeviceSynchronize());

	cv::Mat dst(src.rows, src.cols, src.type());

	// Copy output vector from GPU buffer to host memory.
	ck(cudaMemcpy(dst.data, dev_dst, width * height * sizeof(uint8_t), cudaMemcpyDeviceToHost));

	ck(cudaFree(dev_src));
	ck(cudaFree(dev_dst));

	return dst;
}

cv::Mat GaussianFilterGPUPinned(cv::Mat src)
{
	int width = src.cols, height = src.rows;
	uint8_t *pinned_host;
	uint8_t *dev_src, *dev_dst;

	ck(cudaHostAlloc((void**)&pinned_host, width * height * sizeof(uint8_t), 0));
	memcpy(pinned_host, src.data, width * height * sizeof(uint8_t));

	ck(cudaMalloc((void**)&dev_src, width * height * sizeof(uint8_t)));
	ck(cudaMalloc((void**)&dev_dst, width * height * sizeof(uint8_t)));
	ck(cudaMemcpy(dev_src, pinned_host, width * height * sizeof(uint8_t), cudaMemcpyHostToDevice));

	// Launch a kernel on the GPU with one thread for each element.
	dim3 threadsPerBlock(32, 8);
	dim3 numBlocks(
		(width + threadsPerBlock.x - 1) / threadsPerBlock.x,
		(height + threadsPerBlock.y - 1) / threadsPerBlock.y);
	GaussianKernelSimple << <numBlocks, threadsPerBlock >> >(dev_src, dev_dst, width - 4, height - 4, width);

	// Check for any errors launching the kernel
	ck(cudaGetLastError());

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	ck(cudaDeviceSynchronize());

	cv::Mat dst(src.rows, src.cols, src.type());

	// Copy output vector from GPU buffer to host memory.
	ck(cudaMemcpy(pinned_host, dev_dst, width * height * sizeof(uint8_t), cudaMemcpyDeviceToHost));

	ck(cudaFree(dev_src));
	ck(cudaFree(dev_dst));

	memcpy(dst.data, pinned_host, width * height * sizeof(uint8_t));
	ck(cudaFreeHost(pinned_host));

	return dst;
}

cv::Mat GaussianFilterGPUMapped(cv::Mat src)
{
	int width = src.cols, height = src.rows;
	uint8_t *host_src, *host_dst;

	ck(cudaHostAlloc((void**)&host_src, width * height * sizeof(uint8_t), cudaHostAllocMapped));
	ck(cudaHostAlloc((void**)&host_dst, width * height * sizeof(uint8_t), cudaHostAllocMapped));
	memcpy(host_src, src.data, width * height * sizeof(uint8_t));

	// Launch a kernel on the GPU with one thread for each element.
	dim3 threadsPerBlock(32, 8);
	dim3 numBlocks(
		(width + threadsPerBlock.x - 1) / threadsPerBlock.x,
		(height + threadsPerBlock.y - 1) / threadsPerBlock.y);
	GaussianKernelSimple << <numBlocks, threadsPerBlock >> >(host_src, host_dst, width - 4, height - 4, width);

	// Check for any errors launching the kernel
	ck(cudaGetLastError());

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	ck(cudaDeviceSynchronize());

	cv::Mat dst(src.rows, src.cols, src.type());
	memcpy(dst.data, host_dst, width * height * sizeof(uint8_t));
	ck(cudaFreeHost(host_src));
	ck(cudaFreeHost(host_dst));

	return dst;
}

cv::Mat GaussianFilterGPUMappedOut(cv::Mat src)
{
	int width = src.cols, height = src.rows;
	uint8_t *pinned_host, *mapped_dst;
	uint8_t *dev_src;

	ck(cudaHostAlloc((void**)&pinned_host, width * height * sizeof(uint8_t), cudaHostAllocMapped));
	ck(cudaHostAlloc((void**)&mapped_dst, width * height * sizeof(uint8_t), cudaHostAllocMapped));
	memcpy(pinned_host, src.data, width * height * sizeof(uint8_t));

	ck(cudaMalloc((void**)&dev_src, width * height * sizeof(uint8_t)));
	ck(cudaMemcpy(dev_src, pinned_host, width * height * sizeof(uint8_t), cudaMemcpyHostToDevice));

	// Launch a kernel on the GPU with one thread for each element.
	dim3 threadsPerBlock(32, 8);
	dim3 numBlocks(
		(width + threadsPerBlock.x - 1) / threadsPerBlock.x,
		(height + threadsPerBlock.y - 1) / threadsPerBlock.y);
	GaussianKernelSimple << <numBlocks, threadsPerBlock >> >(dev_src, mapped_dst, width - 4, height - 4, width);

	// Check for any errors launching the kernel
	ck(cudaGetLastError());

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	ck(cudaDeviceSynchronize());

	cv::Mat dst(src.rows, src.cols, src.type());

	ck(cudaFree(dev_src));

	memcpy(dst.data, mapped_dst, width * height * sizeof(uint8_t));
	ck(cudaFreeHost(pinned_host));
	ck(cudaFreeHost(mapped_dst));

	return dst;
}

cv::Mat GaussianFilterGPUManaged(cv::Mat src)
{
	int width = src.cols, height = src.rows;
	uint8_t *host_src, *host_dst;

	ck(cudaMallocManaged((void**)&host_src, width * height * sizeof(uint8_t)));
	ck(cudaMallocManaged((void**)&host_dst, width * height * sizeof(uint8_t)));
	memcpy(host_src, src.data, width * height * sizeof(uint8_t));

	// Launch a kernel on the GPU with one thread for each element.
	dim3 threadsPerBlock(32, 8);
	dim3 numBlocks(
		(width + threadsPerBlock.x - 1) / threadsPerBlock.x,
		(height + threadsPerBlock.y - 1) / threadsPerBlock.y);
	GaussianKernelSimple << <numBlocks, threadsPerBlock >> >(host_src, host_dst, width - 4, height - 4, width);

	// Check for any errors launching the kernel
	ck(cudaGetLastError());

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	ck(cudaDeviceSynchronize());

	cv::Mat dst(src.rows, src.cols, src.type());
	memcpy(dst.data, host_dst, width * height * sizeof(uint8_t));
	ck(cudaFree(host_src));
	ck(cudaFree(host_dst));

	return dst;
}

__global__ void GaussianKernelColor3(const uchar3 *src, uchar3 *dst, int width, int height, int step)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height) {
		float3 sum = { 0, 0, 0 };
		for (int dy = 0; dy < 5; ++dy) {
			for (int dx = 0; dx < 5; ++dx) {
				auto s = src[(x + dx) + (y + dy) * step];
				sum.x += filter[dy][dx] * s.x;
				sum.y += filter[dy][dx] * s.y;
				sum.z += filter[dy][dx] * s.z;
			}
		}
		uchar3 t = { (int)(sum.x + 0.5),(int)(sum.y + 0.5),(int)(sum.z + 0.5) };
		dst[x + y * step] = t;
	}
}

__global__ void GaussianKernelColor4(const uchar4 *src, uchar4 *dst, int width, int height, int step)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height) {
		float3 sum = { 0, 0, 0 };
		for (int dy = 0; dy < 5; ++dy) {
			for (int dx = 0; dx < 5; ++dx) {
				auto s = src[(x + dx) + (y + dy) * step];
				sum.x += filter[dy][dx] * s.x;
				sum.y += filter[dy][dx] * s.y;
				sum.z += filter[dy][dx] * s.z;
			}
		}
		uchar4 t = { (int)(sum.x + 0.5),(int)(sum.y + 0.5),(int)(sum.z + 0.5),0 };
		dst[x + y * step] = t;
	}
}

cv::Mat GaussianFilterGPUColor3(cv::Mat src)
{
	int width = src.cols, height = src.rows;
	uchar3 *dev_src, *dev_dst;

	ck(cudaMalloc((void**)&dev_src, width * height * sizeof(uchar3)));
	ck(cudaMalloc((void**)&dev_dst, width * height * sizeof(uchar3)));
	ck(cudaMemcpy(dev_src, src.data, width * height * sizeof(uchar3), cudaMemcpyHostToDevice));

	// Launch a kernel on the GPU with one thread for each element.
	dim3 threadsPerBlock(32, 8);
	dim3 numBlocks(
		(width + threadsPerBlock.x - 1) / threadsPerBlock.x,
		(height + threadsPerBlock.y - 1) / threadsPerBlock.y);
	GaussianKernelColor3 << <numBlocks, threadsPerBlock >> >(dev_src, dev_dst, width - 4, height - 4, width);

	// Check for any errors launching the kernel
	ck(cudaGetLastError());

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	ck(cudaDeviceSynchronize());

	cv::Mat dst(src.rows, src.cols, src.type());

	// Copy output vector from GPU buffer to host memory.
	ck(cudaMemcpy(dst.data, dev_dst, width * height * sizeof(uchar3), cudaMemcpyDeviceToHost));

	ck(cudaFree(dev_src));
	ck(cudaFree(dev_dst));

	return dst;
}

cv::Mat GaussianFilterGPUColor4(cv::Mat src)
{
	int width = src.cols, height = src.rows;
	uchar4 *dev_src, *dev_dst;

	ck(cudaMalloc((void**)&dev_src, width * height * sizeof(uchar4)));
	ck(cudaMalloc((void**)&dev_dst, width * height * sizeof(uchar4)));
	ck(cudaMemcpy(dev_src, src.data, width * height * sizeof(uchar4), cudaMemcpyHostToDevice));

	// Launch a kernel on the GPU with one thread for each element.
	dim3 threadsPerBlock(32, 8);
	dim3 numBlocks(
		(width + threadsPerBlock.x - 1) / threadsPerBlock.x,
		(height + threadsPerBlock.y - 1) / threadsPerBlock.y);
	GaussianKernelColor4 << <numBlocks, threadsPerBlock >> >(dev_src, dev_dst, width - 4, height - 4, width);

	// Check for any errors launching the kernel
	ck(cudaGetLastError());

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	ck(cudaDeviceSynchronize());

	cv::Mat dst(src.rows, src.cols, src.type());

	// Copy output vector from GPU buffer to host memory.
	ck(cudaMemcpy(dst.data, dev_dst, width * height * sizeof(uchar4), cudaMemcpyDeviceToHost));

	ck(cudaFree(dev_src));
	ck(cudaFree(dev_dst));

	return dst;
}

__global__ void TransposeKernelSimple(const uint8_t *src, uint8_t *dst, int width, int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height)
		dst[y + x * height] = src[x + y * width];
}

__global__ void TransposeKernelShared(const uint8_t *src, uint8_t *dst, int width, int height)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int xbase = blockIdx.x * blockDim.x;
	int ybase = blockIdx.y * blockDim.y;

	__shared__ uint8_t sbuf[16][16];

	{
		int x = xbase + tx;
		int y = ybase + ty;
		if (x < width && y < height)
			sbuf[ty][tx] = src[x + y * width];
	}

	__syncthreads();

	{
		int x = xbase + ty;
		int y = ybase + tx;
		if (x < width && y < height)
			dst[y + x * height] = sbuf[tx][ty];
	}
}

__global__ void TransposeKernelFast(const uint8_t *src, uint8_t *dst, int width, int height)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int xbase = blockIdx.x * blockDim.x;
	int ybase = blockIdx.y * blockDim.y;

	__shared__ uint8_t sbuf[16][16+4];

	{
		int x = xbase + tx;
		int y = ybase + ty;
		if (x < width && y < height)
			sbuf[ty][tx] = src[x + y * width];
	}

	__syncthreads();

	{
		int x = xbase + ty;
		int y = ybase + tx;
		if (x < width && y < height)
			dst[y + x * height] = sbuf[tx][ty];
	}
}

__global__ void TransposeKernelFast2(const uint8_t *src, uint8_t *dst, int width, int height)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int xbase = blockIdx.x * 32;
	int ybase = blockIdx.y * 32;

	__shared__ uint8_t sbuf[32][32+4];

	{
		int x = xbase + tx;
		if (x < width) {
			int yend = min(ybase + 32, height);
			for (int tyy = ty, y = ybase + ty; y < yend; tyy += 8, y += 8) {
				sbuf[tyy][tx] = src[x + y * width];
			}
		}
	}

	__syncthreads();

	{
		int y = ybase + tx;
		if (y < height) {
			int xend = min(xbase + 32, width);
			for (int tyy = ty, x = xbase + ty; x < xend; tyy += 8, x += 8) {
				dst[y + x * height] = sbuf[tx][tyy];
			}
		}
	}
}

cv::Mat TransposeGPU(cv::Mat src, int opt)
{
	int width = src.cols, height = src.rows;
	uint8_t *dev_src, *dev_dst;

	ck(cudaMalloc((void**)&dev_src, width * height * sizeof(uint8_t)));
	ck(cudaMalloc((void**)&dev_dst, width * height * sizeof(uint8_t)));
	ck(cudaMemcpy(dev_src, src.data, width * height * sizeof(uint8_t), cudaMemcpyHostToDevice));

	// Launch a kernel on the GPU with one thread for each element.
	dim3 threadsPerBlock(16, 16);
	dim3 numBlocks(
		(width + threadsPerBlock.x - 1) / threadsPerBlock.x,
		(height + threadsPerBlock.y - 1) / threadsPerBlock.y);
	if(opt == 1)
		TransposeKernelSimple << <numBlocks, threadsPerBlock >> >(dev_src, dev_dst, width, height);
	if (opt == 2)
		TransposeKernelShared << <numBlocks, threadsPerBlock >> >(dev_src, dev_dst, width, height);
	if (opt == 3)
		TransposeKernelFast << <numBlocks, threadsPerBlock >> >(dev_src, dev_dst, width, height);

	// Check for any errors launching the kernel
	ck(cudaGetLastError());

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	ck(cudaDeviceSynchronize());

	cv::Mat dst(src.cols, src.rows, src.type());

	// Copy output vector from GPU buffer to host memory.
	ck(cudaMemcpy(dst.data, dev_dst, width * height * sizeof(uint8_t), cudaMemcpyDeviceToHost));

	ck(cudaFree(dev_src));
	ck(cudaFree(dev_dst));

	return dst;
}

cv::Mat TransposeGPUFast(cv::Mat src)
{
	int width = src.cols, height = src.rows;
	uint8_t *dev_src, *dev_dst;

	ck(cudaMalloc((void**)&dev_src, width * height * sizeof(uint8_t)));
	ck(cudaMalloc((void**)&dev_dst, width * height * sizeof(uint8_t)));
	ck(cudaMemcpy(dev_src, src.data, width * height * sizeof(uint8_t), cudaMemcpyHostToDevice));

	// Launch a kernel on the GPU with one thread for each element.
	dim3 threadsPerBlock(32, 8);
	dim3 numBlocks(
		(width + 32 - 1) / 32,
		(height + 32 - 1) / 32);
	TransposeKernelFast2 << <numBlocks, threadsPerBlock >> >(dev_src, dev_dst, width, height);

	// Check for any errors launching the kernel
	ck(cudaGetLastError());

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	ck(cudaDeviceSynchronize());

	cv::Mat dst(src.cols, src.rows, src.type());

	// Copy output vector from GPU buffer to host memory.
	ck(cudaMemcpy(dst.data, dev_dst, width * height * sizeof(uint8_t), cudaMemcpyDeviceToHost));

	ck(cudaFree(dev_src));
	ck(cudaFree(dev_dst));

	return dst;
}

__global__ void ReduceHKernelSimple(const uint8_t *src, float *dst, int width, int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (x < width) {
		float sum = 0;
		for (int y = 0; y < height; ++y) {
			sum += src[x + y * width];
		}
		dst[x] = sum;
	}
}

__global__ void ReduceInitKernel(float *dst, int length)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (x < length) {
		dst[x] = 0;
	}
}

__global__ void ReduceHKernelFast(const uint8_t *src, float *dst, int width, int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * 128;

	if (x < width) {
		float sum = 0;
		for (int yend = min(y + 128, height); y < yend; ++y) {
			sum += src[x + y * width];
		}
		atomicAdd(&dst[x], sum);
	}
}

cv::Mat ReduceHSimple(cv::Mat src)
{
	int width = src.cols, height = src.rows;
	uint8_t *dev_src;
	float *dev_dst;

	ck(cudaMalloc((void**)&dev_src, width * height * sizeof(uint8_t)));
	ck(cudaMalloc((void**)&dev_dst, width * sizeof(float)));
	ck(cudaMemcpy(dev_src, src.data, width * height * sizeof(uint8_t), cudaMemcpyHostToDevice));

	// Launch a kernel on the GPU with one thread for each element.
	dim3 threadsPerBlock(1024);
	dim3 numBlocks(
		(width + threadsPerBlock.x - 1) / threadsPerBlock.x);
	ReduceHKernelSimple << <numBlocks, threadsPerBlock >> >(dev_src, dev_dst, width, height);

	// Check for any errors launching the kernel
	ck(cudaGetLastError());

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	ck(cudaDeviceSynchronize());

	cv::Mat dst(width, 1, CV_32FC1);

	// Copy output vector from GPU buffer to host memory.
	ck(cudaMemcpy(dst.data, dev_dst, width * sizeof(float), cudaMemcpyDeviceToHost));

	ck(cudaFree(dev_src));
	ck(cudaFree(dev_dst));

	return dst;
}

cv::Mat ReduceHFast(cv::Mat src)
{
	int width = src.cols, height = src.rows;
	uint8_t *dev_src;
	float *dev_dst;

	ck(cudaMalloc((void**)&dev_src, width * height * sizeof(uint8_t)));
	ck(cudaMalloc((void**)&dev_dst, width * sizeof(float)));
	ck(cudaMemcpy(dev_src, src.data, width * height * sizeof(uint8_t), cudaMemcpyHostToDevice));

	{
		dim3 threadsPerBlock(1024);
		dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x);
		ReduceInitKernel << <numBlocks, threadsPerBlock >> >(dev_dst, width);
	}
	{
		dim3 threadsPerBlock(1024);
		dim3 numBlocks(
			(width + threadsPerBlock.x - 1) / threadsPerBlock.x,
			(height + 128 - 1) / 128);
		ReduceHKernelFast << <numBlocks, threadsPerBlock >> >(dev_src, dev_dst, width, height);
	}

	// Check for any errors launching the kernel
	ck(cudaGetLastError());

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	ck(cudaDeviceSynchronize());

	cv::Mat dst(width, 1, CV_32FC1);

	// Copy output vector from GPU buffer to host memory.
	ck(cudaMemcpy(dst.data, dev_dst, width * sizeof(float), cudaMemcpyDeviceToHost));

	ck(cudaFree(dev_src));
	ck(cudaFree(dev_dst));

	return dst;
}

__global__ void ReduceWKernelSimple(const uint8_t *src, float *dst, int width, int height)
{
	int y = blockIdx.x * blockDim.x + threadIdx.x;
	int x = blockIdx.y * 128;

	if (y < height) {
		float sum = 0;
		for (int xend = min(x + 128, width); x < xend; ++x) {
			sum += src[x + y * width];
		}
		atomicAdd(&dst[y], sum);
	}
}

__device__ float ReduceFunc(int tid, float* buf)
{
	if (tid < 256) {
		buf[tid] += buf[tid + 256];
	}
	__syncthreads();
	if (tid < 128) {
		buf[tid] += buf[tid + 128];
	}
	__syncthreads();
	if (tid < 64) {
		buf[tid] += buf[tid + 64];
	}
	__syncthreads();
	float sum;
	if (tid < 32) {
		sum = buf[tid] + buf[tid + 32];
		sum += __shfl_down_sync(0xffffffff, sum, 16);
		sum += __shfl_down_sync(0xffffffff, sum, 8);
		sum += __shfl_down_sync(0xffffffff, sum, 4);
		sum += __shfl_down_sync(0xffffffff, sum, 2);
		sum += __shfl_down_sync(0xffffffff, sum, 1);
	}
	return sum;
}

__global__ void ReduceWKernelFast(const uint8_t *src, float *dst, int width, int height)
{
	int tid = threadIdx.x;
	int y = blockIdx.y;

	__shared__ float sbuf[512];

	float sum = 0;
	for (int x = tid; x < width; x += 512) {
		sum += src[x + y * width];
	}

	sbuf[tid] = sum;
	__syncthreads();

	sum = ReduceFunc(tid, sbuf);

	if (tid == 0)
		dst[y] = sum;
}

cv::Mat ReduceWSimple(cv::Mat src)
{
	int width = src.cols, height = src.rows;
	uint8_t *dev_src;
	float *dev_dst;

	ck(cudaMalloc((void**)&dev_src, width * height * sizeof(uint8_t)));
	ck(cudaMalloc((void**)&dev_dst, height * sizeof(float)));
	ck(cudaMemcpy(dev_src, src.data, width * height * sizeof(uint8_t), cudaMemcpyHostToDevice));

	{
		dim3 threadsPerBlock(1024);
		dim3 numBlocks((height + threadsPerBlock.x - 1) / threadsPerBlock.x);
		ReduceInitKernel << <numBlocks, threadsPerBlock >> >(dev_dst, height);
	}
	{
		dim3 threadsPerBlock(1024);
		dim3 numBlocks(
			(height + threadsPerBlock.x - 1) / threadsPerBlock.x,
			(width + 128 - 1) / 128);
		ReduceWKernelSimple << <numBlocks, threadsPerBlock >> >(dev_src, dev_dst, width, height);
	}

	// Check for any errors launching the kernel
	ck(cudaGetLastError());

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	ck(cudaDeviceSynchronize());

	cv::Mat dst(height, 1, CV_32FC1);

	// Copy output vector from GPU buffer to host memory.
	ck(cudaMemcpy(dst.data, dev_dst, height * sizeof(float), cudaMemcpyDeviceToHost));

	ck(cudaFree(dev_src));
	ck(cudaFree(dev_dst));

	return dst;
}

cv::Mat ReduceWFast(cv::Mat src)
{
	int width = src.cols, height = src.rows;
	uint8_t *dev_src;
	float *dev_dst;

	ck(cudaMalloc((void**)&dev_src, width * height * sizeof(uint8_t)));
	ck(cudaMalloc((void**)&dev_dst, height * sizeof(float)));
	ck(cudaMemcpy(dev_src, src.data, width * height * sizeof(uint8_t), cudaMemcpyHostToDevice));

	// Launch a kernel on the GPU with one thread for each element.
	dim3 threadsPerBlock(512);
	dim3 numBlocks(height);
	ReduceWKernelFast << <numBlocks, threadsPerBlock >> >(dev_src, dev_dst, width, height);

	// Check for any errors launching the kernel
	ck(cudaGetLastError());

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	ck(cudaDeviceSynchronize());

	cv::Mat dst(height, 1, CV_32FC1);

	// Copy output vector from GPU buffer to host memory.
	ck(cudaMemcpy(dst.data, dev_dst, height * sizeof(float), cudaMemcpyDeviceToHost));

	ck(cudaFree(dev_src));
	ck(cudaFree(dev_dst));

	return dst;
}
