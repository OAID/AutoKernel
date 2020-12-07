#include "HalideBuffer.h"
#include <iostream>
#include "halide_depthwise.h"

int main(int argc, char **argv)
{
    int C = 2, W = 4, H = 4, N = 1;
    Halide::Runtime::Buffer<float> input_tensor(nullptr, W, H, C, N);
    Halide::Runtime::Buffer<float> output_tensor(nullptr, W, H, C, N);
    Halide::Runtime::Buffer<float> filter(nullptr, 3, 3, C, N);
    Halide::Runtime::Buffer<float> bias(nullptr, C);
    input_tensor.allocate();
    output_tensor.allocate();
    filter.allocate();
    bias.allocate();
    input_tensor.for_each_value([](float &x) {
        x = (int)((2.0 * rand() / RAND_MAX) * 10)  - 1.0;
    });

    output_tensor.for_each_value([](float &x) {
    	x = (int)((2.0 * rand() / RAND_MAX) * 10)  - 1.0;
    });
    filter.for_each_value([](float &x) {
    	x = (int)((2.0 * rand() / RAND_MAX) * 10)  - 1.0;
    });
    bias.for_each_value([](float &x){
    	x = (int)((2.0 * rand() / RAND_MAX) * 10)  - 1.0;
    });

    halide_depthwise(input_tensor, filter, bias, 1, 1, 1, output_tensor);

    printf("input:\n");
    for (int c = 0; c < input_tensor.dim(3).extent(); c++) {
        for (int z = 0; z < input_tensor.channels(); z++) {
            for (int y = 0; y < input_tensor.height(); y++) {
                for (int x = 0; x < input_tensor.width(); x++) {
                    std::cout<<input_tensor(x,y,z,0)<<" ";
                }
                std::cout<<"\n";
            }
            std::cout<<"\n";
        }
    }
    printf("filter:\n");
    for (int c = 0; c < filter.dim(3).extent(); c++) {
        for (int z = 0; z < filter.channels(); z++) {
            for (int y = 0; y < filter.height(); y++) {
                for (int x = 0; x < filter.width(); x++) {
                    std::cout<<filter(x,y,z,0)<<" ";
                }
                std::cout<<"\n";
            }
            std::cout<<"\n";
        }
    }
    std::cout << "bias = " << bias(0) << " , " << bias(1) << " ." << std::endl;
    std::cout  << "stride = " << 1  << " pad_width = " << 1 << " pad_height = " << 1 << "\n";
    
    printf("output:\n");
    for (int c = 0; c < output_tensor.dim(3).extent(); c++) {
        for (int z = 0; z < output_tensor.channels(); z++) {
            for (int y = 0; y < output_tensor.height(); y++) {
                for (int x = 0; x < output_tensor.width(); x++) {
                    std::cout<<output_tensor(x,y,z,0)<<" ";
                }
                std::cout<<"\n";
            }
            std::cout<<"\n";
        }
    }

    return 0;
}
