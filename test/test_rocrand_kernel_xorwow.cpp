// Copyright (c) 2017-2021 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include <stdio.h>
#include <gtest/gtest.h>

#include <vector>
#include <cmath>
#include <type_traits>

#include <hip/hip_runtime.h>

#define FQUALIFIERS __forceinline__ __host__ __device__
#include <rocrand/rocrand_kernel.h>
#include <rocrand/rocrand.h>

#include "test_common.hpp"
#include "test_rocrand_common.hpp"


template <class GeneratorState>
__global__
// __launch_bounds__(64) // Causes errors on MI200/HIP on Windows gfx1030
void rocrand_poisson_kernel(unsigned int * output, const size_t size, double lambda)
{
    const unsigned int state_id = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const unsigned int global_size = hipGridDim_x * hipBlockDim_x;

    GeneratorState state;
    const unsigned int subsequence = state_id;
    rocrand_init(0, subsequence, 234ULL, &state);

    unsigned int index = state_id;
    while(index < size)
    {
        output[index] = rocrand_poisson(&state, lambda);
        index += global_size;
    }
}


class rocrand_kernel_xorwow_poisson : public ::testing::TestWithParam<double> { };

TEST_P(rocrand_kernel_xorwow_poisson, rocrand_poisson)
{
    typedef rocrand_state_xorwow state_type;

    const double lambda = GetParam();

    const size_t output_size = 8192;
    unsigned int * output;
    HIP_CHECK(hipMallocHelper((void **)&output, output_size * sizeof(unsigned int)));
    HIP_CHECK(hipDeviceSynchronize());

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(rocrand_poisson_kernel<state_type>),
        dim3(4), dim3(64), 0, 0,
        output, output_size, lambda
    );
    HIP_CHECK(hipGetLastError());

    std::vector<unsigned int> output_host(output_size);
    HIP_CHECK(
        hipMemcpy(
            output_host.data(), output,
            output_size * sizeof(unsigned int),
            hipMemcpyDeviceToHost
        )
    );
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(output));

    double mean = 0;
    for(auto v : output_host)
    {
        mean += static_cast<double>(v);
    }
    mean = mean / output_size;

    double variance = 0;
    for(auto v : output_host)
    {
        variance += std::pow(v - mean, 2);
    }
    variance = variance / output_size;

    EXPECT_NEAR(mean, lambda, std::max(1.0, lambda * 1e-1));
    EXPECT_NEAR(variance, lambda, std::max(1.0, lambda * 1e-1));
}

const double lambdas[] = { 1.0, 5.5 };

INSTANTIATE_TEST_SUITE_P(rocrand_kernel_xorwow_poisson,
                        rocrand_kernel_xorwow_poisson,
                        ::testing::ValuesIn(lambdas));
