#include "HalideBuffer.h"
#include "HalideRuntimeCuda.h"
#include "halide_benchmark.h"
#include "vector"
#include "iostream"
using Halide::Runtime::Buffer;
using Halide::Tools::benchmark;
using namespace std;
void init(Buffer<float> &B)
{
    for (auto iter=B.begin();iter!=B.end();iter++)
    {
        (*iter) = rand()*1.0/RAND_MAX;
    }
}
int main(int argc, char **argv) {
    {
        INPUT_TEMPLATE;
        INIT_INPUT;
        const auto benchmark_inner = [&]() {
            FUNC(DEMO_ARGS);
            OUTPUT.device_sync();
        };
        double t  = Halide::Tools::benchmark(SAMPLES,ITERATORS,benchmark_inner);
        std::cout<<"autokernel time:\t"<<t*1000<<" ms\n";
    }
    return 0;
}
