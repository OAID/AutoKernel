#include <unistd.h>
#include <iostream>
#include <string>
#include <algorithm>

#include "tengine_c_api.h"
#include "tengine_operations.h"

const char* model_file = "squeezenet.tmfile";
const char* image_file = "cat.jpg";

using namespace std;

int main()
{
    // check files
    if(!check_file_exist(model_file) || !check_file_exist(image_file))
    {
        return -1;
    }

    int img_h = 227;
    int img_w = 227;
    float mean[3] = {104.007, 116.669, 122.679};
    float scale[3] = {1.f, 1.f, 1.f};

    /* set runtime options of Net */
    struct options opt;
    opt.num_thread = 1;
    opt.precision = TENGINE_MODE_FP32;
    opt.cluster = TENGINE_CLUSTER_ALL;

    /* load model */
    init_tengine();
    graph_t graph = create_graph(NULL, "tengine", model_file);

    /* prepare input data */
    int img_size = img_h * img_w * 3;
    int dims[] = {1, 3, img_h, img_w}; 
    float* input_data = ( float* )malloc(img_size * sizeof(float));
    get_input_data(image_file, input_data, img_h, img_w, mean, scale);
    tensor_t input_tensor = get_graph_input_tensor(graph, 0, 0);
    set_tensor_shape(input_tensor, dims, 4);
    set_tensor_buffer(input_tensor, input_data, img_size * sizeof(float));

    /* forward */
    prerun_graph_multithread(graph, opt);
    run_graph(graph, 1);

    /* get result */
    tensor_t output_tensor = get_graph_output_tensor(graph, 0, 0);
    float* output_data = ( float* )get_tensor_buffer(output_tensor);
    int output_size = get_tensor_buffer_size(output_tensor) / sizeof(float);

    /* after process */
    print_topk(output_data, output_size, 5);
    std::cout << "--------------------------------------\n";
    std::cout << "ALL TEST DONE\n";


    free(input_data);
    postrun_graph(graph);
    destroy_graph(graph);
    release_tengine();
    return 0;
}
