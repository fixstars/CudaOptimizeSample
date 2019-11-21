#include "gtest/gtest.h"
#include <chrono>

#ifdef _DEBUG
#define GTEST_SUFFIX_DBG_STR "d"
#else
#define GTEST_SUFFIX_DBG_STR ""
#endif
#pragma comment (lib, "gtest" GTEST_SUFFIX_DBG_STR ".lib")

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#define	CV_VERSION_STR	CVAUX_STR(CV_MAJOR_VERSION)	CVAUX_STR(CV_MINOR_VERSION)	CVAUX_STR(CV_SUBMINOR_VERSION)
#ifdef _DEBUG
#define CV_SUFFIX_DBG_STR "d"
#else
#define CV_SUFFIX_DBG_STR ""
#endif
#pragma	comment	(lib, "opencv_"	"core"	CV_VERSION_STR	CV_SUFFIX_DBG_STR	".lib")
#pragma	comment	(lib, "opencv_"	"imgcodecs"	CV_VERSION_STR	CV_SUFFIX_DBG_STR	".lib")
#pragma	comment	(lib, "opencv_"	"imgproc"	CV_VERSION_STR	CV_SUFFIX_DBG_STR	".lib")

void GaussianKernelCPU(const uint8_t *src, uint8_t *dst, int width, int height, int step);
cv::Mat GaussianFilterCPUMT(cv::Mat src);
cv::Mat GaussianFilterGPUSimple(cv::Mat src);
cv::Mat GaussianFilterGPUOpt(cv::Mat src, int opt);
cv::Mat GaussianFilterGPUArray(cv::Mat src);
cv::Mat GaussianFilterGPUPinned(cv::Mat src);
cv::Mat GaussianFilterGPUMapped(cv::Mat src);
cv::Mat GaussianFilterGPUMappedOut(cv::Mat src);
cv::Mat GaussianFilterGPUManaged(cv::Mat src);
cv::Mat GaussianFilterGPUColor3(cv::Mat src);
cv::Mat GaussianFilterGPUColor4(cv::Mat src);

cv::Mat BilateralFilterGPUOpt(cv::Mat src, int opt);

cv::Mat TransposeGPU(cv::Mat src, int opt);
cv::Mat TransposeGPUFast(cv::Mat src);


void GaussianKernelCPUMT(const uint8_t *src, uint8_t *dst, int width, int height, int step)
{
	const float filter[5][5] = {
		{ 0.002969017f, 0.01330621f, 0.021938231f, 0.01330621f, 0.002969017f },
		{ 0.01330621f, 0.059634295f, 0.098320331f, 0.059634295f, 0.01330621f },
		{ 0.021938231f, 0.098320331f, 0.162102822f, 0.098320331f, 0.021938231f },
		{ 0.01330621f, 0.059634295f, 0.098320331f, 0.059634295f, 0.01330621f },
		{ 0.002969017f, 0.01330621f, 0.021938231f, 0.01330621f, 0.002969017f },
	};
#pragma omp parallel for
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

cv::Mat GaussianFilterCPU(cv::Mat src)
{
	int width = src.cols, height = src.rows;
	cv::Mat dst(src.rows, src.cols, src.type());
	auto start = std::chrono::system_clock::now();
	GaussianKernelCPU(src.data, dst.data, width - 4, height - 4, width);
	auto end = std::chrono::system_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	printf("GaussianFilterCPU: %lld ms\n", elapsed);
	return dst;
}

cv::Mat GaussianFilterCPUMT(cv::Mat src)
{
	int width = src.cols, height = src.rows;
	cv::Mat dst(src.rows, src.cols, src.type());
	auto start = std::chrono::system_clock::now();
	GaussianKernelCPUMT(src.data, dst.data, width - 4, height - 4, width);
	auto end = std::chrono::system_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	printf("GaussianFilterCPUMT: %lld ms\n", elapsed);
	return dst;
}

TEST(Original, Gray) {
	cv::Mat src = cv::imread("../sample.jpg", cv::IMREAD_GRAYSCALE);
	cv::imwrite("../out/gray.png", src
	);
}

TEST(Gaussian, CPU) {
	cv::Mat src = cv::imread("../sample.jpg", cv::IMREAD_GRAYSCALE);
	cv::Mat dst = GaussianFilterCPU(src);
	cv::imwrite("../out/gaussian_cpu.png", dst);
}

TEST(Gaussian, CPUMT) {
	cv::Mat src = cv::imread("../sample.jpg", cv::IMREAD_GRAYSCALE);
	cv::Mat dst = GaussianFilterCPUMT(src);
	cv::imwrite("../out/gaussian_cpu_mt.png", dst);
}

TEST(Gaussian, Simple) {
	cv::Mat src = cv::imread("../sample.jpg", cv::IMREAD_GRAYSCALE);
	cv::Mat dst = GaussianFilterGPUSimple(src);
	cv::imwrite("../out/gaussian_simple.png", dst);
}

TEST(Gaussian, Array) {
	cv::Mat src = cv::imread("../sample.jpg", cv::IMREAD_GRAYSCALE);
	cv::Mat dst = GaussianFilterGPUOpt(src, 1);
	cv::imwrite("../out/gaussian_array.png", dst);
}

TEST(Gaussian, Constant) {
	cv::Mat src = cv::imread("../sample.jpg", cv::IMREAD_GRAYSCALE);
	cv::Mat dst = GaussianFilterGPUOpt(src, 2);
	cv::imwrite("../out/gaussian_copnstant.png", dst);
}

TEST(Gaussian, ConstantFixed) {
	cv::Mat src = cv::imread("../sample.jpg", cv::IMREAD_GRAYSCALE);
	cv::Mat dst = GaussianFilterGPUOpt(src, 3);
	cv::imwrite("../out/gaussian_constant_fixed.png", dst);
}

TEST(Gaussian, Shared) {
	cv::Mat src = cv::imread("../sample.jpg", cv::IMREAD_GRAYSCALE);
	cv::Mat dst = GaussianFilterGPUOpt(src, 4);
	cv::imwrite("../out/gaussian_shared.png", dst);
}

TEST(Gaussian, Pinned) {
	cv::Mat src = cv::imread("../sample.jpg", cv::IMREAD_GRAYSCALE);
	cv::Mat dst = GaussianFilterGPUPinned(src);
	cv::imwrite("../out/gaussian_pinned.png", dst);
}

TEST(Gaussian, Mapped) {
	cv::Mat src = cv::imread("../sample.jpg", cv::IMREAD_GRAYSCALE);
	cv::Mat dst = GaussianFilterGPUMapped(src);
	cv::imwrite("../out/gaussian_mapped.png", dst);
}

TEST(Gaussian, MappedOut) {
	cv::Mat src = cv::imread("../sample.jpg", cv::IMREAD_GRAYSCALE);
	cv::Mat dst = GaussianFilterGPUMappedOut(src);
	cv::imwrite("../out/gaussian_mapped_out.png", dst);
}

TEST(Gaussian, Managed) {
	cv::Mat src = cv::imread("../sample.jpg", cv::IMREAD_GRAYSCALE);
	cv::Mat dst = GaussianFilterGPUManaged(src);
	cv::imwrite("../out/gaussian_managed.png", dst);
}

TEST(Gaussian, Color3) {
	cv::Mat src = cv::imread("../sample.jpg", cv::IMREAD_COLOR);
	cv::Mat dst = GaussianFilterGPUColor3(src);
	cv::imwrite("../out/gaussian_color3.png", dst);
}

TEST(Gaussian, Color4) {
	cv::Mat src = cv::imread("../sample.jpg", cv::IMREAD_COLOR);
	cv::Mat src4;
	cv::cvtColor(src, src4, cv::COLOR_BGR2BGRA);
	cv::Mat dst4 = GaussianFilterGPUColor4(src4);
	cv::Mat dst;
	cv::cvtColor(dst4, dst, cv::COLOR_BGRA2BGR);
	cv::imwrite("../out/gaussian_color4.png", dst);
}

TEST(Bilateral, Naive) {
	cv::Mat src = cv::imread("../sample.jpg", cv::IMREAD_GRAYSCALE);
	cv::Mat dst = BilateralFilterGPUOpt(src, 1);
	cv::imwrite("../out/bilateral_naive.png", dst);
}

TEST(Bilateral, Simple) {
	cv::Mat src = cv::imread("../sample.jpg", cv::IMREAD_GRAYSCALE);
	cv::Mat dst = BilateralFilterGPUOpt(src, 2);
	cv::imwrite("../out/bilateral_simple.png", dst);
}

TEST(Bilateral, Fast) {
	cv::Mat src = cv::imread("../sample.jpg", cv::IMREAD_GRAYSCALE);
	cv::Mat dst = BilateralFilterGPUOpt(src, 3);
	cv::imwrite("../out/bilateral_fast.png", dst);
}

TEST(Transpose, Simple) {
	cv::Mat src = cv::imread("../sample.jpg", cv::IMREAD_GRAYSCALE);
	cv::Mat dst = TransposeGPU(src, 1);
	cv::imwrite("../out/transpose_simple.png", dst);
}

TEST(Transpose, Shared) {
	cv::Mat src = cv::imread("../sample.jpg", cv::IMREAD_GRAYSCALE);
	cv::Mat dst = TransposeGPU(src, 2);
	cv::imwrite("../out/transpose_shared.png", dst);
}

TEST(Transpose, ConflictFree) {
	cv::Mat src = cv::imread("../sample.jpg", cv::IMREAD_GRAYSCALE);
	cv::Mat dst = TransposeGPU(src, 3);
	cv::imwrite("../out/transpose_conflict_free.png", dst);
}

TEST(Transpose, Fast) {
	cv::Mat src = cv::imread("../sample.jpg", cv::IMREAD_GRAYSCALE);
	cv::Mat dst = TransposeGPUFast(src);
	cv::imwrite("../out/transpose_fast.png", dst);
}

void ReduceHCPU(const uint8_t *src, float *dst, int width, int height)
{
	for (int x = 0; x < width; ++x) {
		dst[x] = 0;
	}
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			dst[x] += src[x + y * width];
		}
	}
}

void ReduceHCPUMT(const uint8_t *src, float *dst, int width, int height)
{
	for (int x = 0; x < width; ++x) {
		dst[x] = 0;
	}
	for (int y = 0; y < height; ++y) {
#pragma omp parallel for
		for (int x = 0; x < width; ++x) {
			dst[x] += src[x + y * width];
		}
	}
}

void ReduceWCPU(const uint8_t *src, float *dst, int width, int height)
{
	for (int y = 0; y < height; ++y) {
		float sum = 0;
		for (int x = 0; x < width; ++x) {
			sum += src[x + y * width];
		}
		dst[y] = sum;
	}
}

void ReduceWCPUMT(const uint8_t *src, float *dst, int width, int height)
{
#pragma omp parallel for
	for (int y = 0; y < height; ++y) {
		float sum = 0;
		for (int x = 0; x < width; ++x) {
			sum += src[x + y * width];
		}
		dst[y] = sum;
	}
}

cv::Mat ReduceHSimple(cv::Mat src);
cv::Mat ReduceHFast(cv::Mat src);
cv::Mat ReduceWSimple(cv::Mat src);
cv::Mat ReduceWFast(cv::Mat src);

TEST(Reduce, HCPU) {
	cv::Mat src = cv::imread("../sample.jpg", cv::IMREAD_GRAYSCALE);
	cv::Mat ref(src.cols, 1, CV_32FC1);
	auto start = std::chrono::system_clock::now();
	for(int i = 0; i < 10; ++i)
		ReduceHCPU(src.data, (float*)ref.data, src.cols, src.rows);
	auto end = std::chrono::system_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	printf("ReduceHCPU: %.1f ms\n", elapsed / 10.0);
}

TEST(Reduce, HCPUMT) {
	cv::Mat src = cv::imread("../sample.jpg", cv::IMREAD_GRAYSCALE);
	cv::Mat ref(src.cols, 1, CV_32FC1);
	auto start = std::chrono::system_clock::now();
	for (int i = 0; i < 10; ++i)
		ReduceHCPUMT(src.data, (float*)ref.data, src.cols, src.rows);
	auto end = std::chrono::system_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	printf("ReduceHCPUMT: %.1f ms\n", elapsed / 10.0);
}

TEST(Reduce, WCPU) {
	cv::Mat src = cv::imread("../sample.jpg", cv::IMREAD_GRAYSCALE);
	cv::Mat ref(src.rows, 1, CV_32FC1);
	auto start = std::chrono::system_clock::now();
	for (int i = 0; i < 10; ++i)
		ReduceWCPU(src.data, (float*)ref.data, src.cols, src.rows);
	auto end = std::chrono::system_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	printf("ReduceWCPU: %.1f ms\n", elapsed / 10.0);
}

TEST(Reduce, WCPUMT) {
	cv::Mat src = cv::imread("../sample.jpg", cv::IMREAD_GRAYSCALE);
	cv::Mat ref(src.rows, 1, CV_32FC1);
	auto start = std::chrono::system_clock::now();
	for (int i = 0; i < 10; ++i)
		ReduceWCPUMT(src.data, (float*)ref.data, src.cols, src.rows);
	auto end = std::chrono::system_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	printf("ReduceWCPUMT: %.1f ms\n", elapsed / 10.0);
}

TEST(Reduce, HSimple) {
	cv::Mat src = cv::imread("../sample.jpg", cv::IMREAD_GRAYSCALE);
	cv::Mat dst = ReduceHSimple(src);
	cv::Mat ref(dst.cols, dst.rows, dst.type());
	ReduceHCPU(src.data, (float*)ref.data, src.cols, src.rows);
	for (int i = 0; i < dst.cols; ++i) {
		if (std::abs(dst.at<float>(i) - ref.at<float>(i)) >= 1) {
			printf("ERROR %f vs %f\n", dst.at<float>(i), ref.at<float>(i));
			break;
		}
	}
}

TEST(Reduce, HFast) {
	cv::Mat src = cv::imread("../sample.jpg", cv::IMREAD_GRAYSCALE);
	cv::Mat dst = ReduceHFast(src);
	cv::Mat ref(dst.cols, dst.rows, dst.type());
	ReduceHCPU(src.data, (float*)ref.data, src.cols, src.rows);
	for (int i = 0; i < dst.cols; ++i) {
		if (std::abs(dst.at<float>(i) - ref.at<float>(i)) >= 1) {
			printf("ERROR %f vs %f\n", dst.at<float>(i), ref.at<float>(i));
			break;
		}
	}
}

TEST(Reduce, WSimple) {
	cv::Mat src = cv::imread("../sample.jpg", cv::IMREAD_GRAYSCALE);
	cv::Mat dst = ReduceWSimple(src);
	cv::Mat ref(dst.cols, dst.rows, dst.type());
	ReduceWCPU(src.data, (float*)ref.data, src.cols, src.rows);
	for (int i = 0; i < dst.cols; ++i) {
		if (std::abs(dst.at<float>(i) - ref.at<float>(i)) >= 1) {
			printf("ERROR %f vs %f\n", dst.at<float>(i), ref.at<float>(i));
			break;
		}
	}
}

TEST(Reduce, WFast) {
	cv::Mat src = cv::imread("../sample.jpg", cv::IMREAD_GRAYSCALE);
	cv::Mat dst = ReduceWFast(src);
	cv::Mat ref(dst.cols, dst.rows, dst.type());
	ReduceWCPU(src.data, (float*)ref.data, src.cols, src.rows);
	for (int i = 0; i < dst.cols; ++i) {
		if (std::abs(dst.at<float>(i) - ref.at<float>(i)) >= 1) {
			printf("ERROR %f vs %f\n", dst.at<float>(i), ref.at<float>(i));
			break;
		}
	}
}

int main(int argc, char **argv) {
	::testing::InitGoogleTest(&argc, argv);
	auto ret = RUN_ALL_TESTS();
	//getchar();
	return ret;
}
