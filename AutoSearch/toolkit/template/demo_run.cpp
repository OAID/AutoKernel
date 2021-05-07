#include "HalideBuffer.h"
#include "HalideRuntimeCuda.h"
#include "halide_benchmark.h"
#include <chrono>
#include "vector"
#include "iostream"
using Halide::Runtime::Buffer;
using namespace std::chrono;
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
        double benchmark_min_time=0.1f;
        Halide::Tools::BenchmarkConfig config;
        config.min_time = benchmark_min_time;
        config.max_time = benchmark_min_time * 4;
        double total_time = 0.0f;
        steady_clock::time_point start = steady_clock::now();
        for (int i=0;i<SAMPLES;i++)
        {
            //std::cout<<"samples:"<<i<<std::endl;
            for (int j=0;j<ITERATORS;j++)
            {
                FUNC(DEMO_ARGS);
                OUTPUT.device_sync();
            }
            
            
        }
        steady_clock::time_point end = steady_clock::now();
        duration<double> time_span = duration_cast<duration<double>>(end - start);
        //double t  = Halide::Tools::benchmark(SAMPLES,ITERATORS,benchmark_inner);
        std::cout<<"autokernel time:\t"<<time_span.count()*1000.0/(SAMPLES*ITERATORS)<<" ms\n";
    }
    return 0;
}
