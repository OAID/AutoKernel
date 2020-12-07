#include "HalideBuffer.h"
#include <iostream>
#include "halide_fc.h"

int main(int argc, char **argv)
{
    int CI = 10, CO = 5, N = 1;
    Halide::Runtime::Buffer<float> input_tensor(nullptr, CI, N);
    Halide::Runtime::Buffer<float> output_tensor(nullptr, CO, N);
    Halide::Runtime::Buffer<float> filter(nullptr, CI, CO);
    Halide::Runtime::Buffer<float> bias(nullptr, CO);
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

    halide_fc(input_tensor, filter, bias, CI ,output_tensor);

   
    printf("input:\n");
    for (int y = 0; y < input_tensor.height(); y++) {
        for (int x = 0; x < input_tensor.width(); x++) {
            std::cout<<input_tensor(x,y)<<" ";
        }
        std::cout<<"\n";
    }
    std::cout<<"\n";

    printf("filter:\n");
    for (int y = 0; y < filter.height(); y++) {
         for (int x = 0; x < filter.width(); x++) {
             std::cout<<filter(x,y)<<" ";
         }
         std::cout<<"\n";
    }
    std::cout<<"\n";
   
    std::cout << "bias: \n";
    for(int x = 0; x < CO; x++){
        std::cout << bias(x) << " "; 
    }
    std::cout << std::endl;
   
    printf("output:\n");
    for (int y = 0; y < output_tensor.height(); y++) {
         for (int x = 0; x < output_tensor.width(); x++) {
              std::cout<<output_tensor(x,y)<<" ";
         }
         std::cout<<"\n";
    }
    std::cout<<"\n";

    return 0;
}
