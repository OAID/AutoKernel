#include <iostream>
#include <string>
#include "utils.hpp" //is_file_exist
/* the sample code to create a pooling and do calculation */

#include "tengine/c_api.h"

static inline unsigned long get_cur_time(void)
{
    struct timespec tm;

    clock_gettime(CLOCK_MONOTONIC, &tm);

    return (tm.tv_sec * 1000000 + tm.tv_nsec / 1000);
}

int create_input_node(graph_t graph, const char* node_name, int c, int h, int w)
{
    node_t node = create_graph_node(graph, node_name, "InputOp");
    tensor_t tensor = create_graph_tensor(graph, node_name, TENGINE_DT_FP32);
    set_node_output_tensor(node, 0, tensor, TENSOR_TYPE_INPUT);

    // int dims[4] = {1, h, w, c};	//nhwc
	int dims[4] = {2, c, h, w};  //nchw

    set_tensor_shape(tensor, dims, 4);

    release_graph_tensor(tensor);
    release_graph_node(node);

    return 0;
}

int create_pool_node(graph_t graph, const char* node_name, const char* input_name, int k_size, int stride, int pad,
                     int in_c, int pool_method)
{
    node_t pool_node = create_graph_node(graph, node_name, "Pooling");

    tensor_t input_tensor = get_graph_tensor(graph, input_name);

    if(input_tensor == nullptr)
    {
        std::cout << "ERRNO: " << get_tengine_errno() << "\n";
        return -1;
    }

    set_node_input_tensor(pool_node, 0, input_tensor);

    release_graph_tensor(input_tensor);

    /* output */
    tensor_t output_tensor = create_graph_tensor(graph, node_name, TENGINE_DT_FP32);
    set_node_output_tensor(pool_node, 0, output_tensor, TENSOR_TYPE_VAR);

    release_graph_tensor(output_tensor);

    /* attr */
    set_node_attr_int(pool_node, "kernel_h", &k_size);
    set_node_attr_int(pool_node, "kernel_w", &k_size);
    set_node_attr_int(pool_node, "stride_h", &stride);
    set_node_attr_int(pool_node, "stride_w", &stride);
    set_node_attr_int(pool_node, "pad_h0", &pad);
    set_node_attr_int(pool_node, "pad_w0", &pad);
    set_node_attr_int(pool_node, "pad_h1", &pad);
    set_node_attr_int(pool_node, "pad_w1", &pad);
    set_node_attr_int(pool_node, "pool_method", &pool_method);


    release_graph_node(pool_node);

    return 0;
}

graph_t create_pool_graph(int c, int h, int w, int ksize, int stride, int pad, int pool_method)
{
    graph_t graph = create_graph(nullptr, nullptr, nullptr);
    // set_graph_layout(graph, TENGINE_LAYOUT_NHWC);	//nhwc
	set_graph_layout(graph, TENGINE_LAYOUT_NCHW);    //nchw

    if(graph == nullptr)
    {
        std::cerr << "ERRNO: " << get_tengine_errno() << "\n";
        return nullptr;
    }

    const char* input_name = "data";
    const char* pool_name = "pool";

    if(create_input_node(graph, input_name, c, h, w) < 0)
    {
        std::cerr << "create input failed\n";
        return nullptr;
    }

    if(create_pool_node(graph, pool_name, input_name, ksize, stride, pad, c, pool_method) < 0)
    {
        std::cerr << "create pool node failed\n";
        return nullptr;
    }

    /* set input/output node */

    const char* inputs[] = {input_name};
    const char* outputs[] = {pool_name};

    if(set_graph_input_node(graph, inputs, sizeof(inputs) / sizeof(char*)) < 0)
    {
        std::cerr << "set inputs failed: ERRNO: " << get_tengine_errno() << "\n";
        return nullptr;
    }

    if(set_graph_output_node(graph, outputs, sizeof(outputs) / sizeof(char*)) < 0)
    {
        std::cerr << "set outputs failed: ERRNO: " << get_tengine_errno() << "\n";
        return nullptr;
    }

    return graph;
}

int test_pool(int in_c, int h, int w, int ksize, int stride, int pad, int pool_method)
{
    graph_t graph = create_pool_graph(in_c, h, w, ksize, stride, pad, pool_method);

    if(graph == nullptr)
        return 1;

    /* set input */
    tensor_t input_tensor = get_graph_input_tensor(graph, 0, 0);

    int buf_size = get_tensor_buffer_size(input_tensor);
    float* i_buf = ( float* )malloc(buf_size);
    srand(10);
    for(unsigned int i = 0; i < buf_size / sizeof(float); i++)
    {
        // i_buf[i] = i%10;//rand() % 10;
        i_buf[i] = rand() % 10;
    }

    set_tensor_buffer(input_tensor, i_buf, buf_size);
    release_graph_tensor(input_tensor);

    
    node_t pool_node = get_graph_node(graph, "pool");
 
    // prerun graph
    if(prerun_graph(graph) < 0)
    {
        std::cerr << "prerun_graph failed: ERRNO: " << get_tengine_errno() << "\n";
        return 1;
    }

    int repeat_count = 1;
    printf("REPEAT COUNT= %d\n", repeat_count);
    unsigned long start_time = get_cur_time();

    for(int i = 0; i < repeat_count; i++)
        run_graph(graph, 1);

    unsigned long end_time = get_cur_time();

    unsigned long off_time = end_time - start_time;

    std::printf("Repeat [%d] time %.2f us per RUN. used %lu us\n", repeat_count, 1.0f * off_time / repeat_count, off_time);

    tensor_t output_tensor = get_node_output_tensor(pool_node, 0);

    float* buf = ( float* )get_tensor_buffer(output_tensor);
    // int size = get_tensor_buffer_size(output_tensor);
    // if(float_mismatch(buf, buf1, size/sizeof(float)) != 0)
    //     printf("test failed\n");
    std::cout<<"print output data\n";
    for(int i = 0; i < 50; i++)
    {
        std::cout<<buf[i]<<" ";
    }
    std::cout<<"\n";
    /*
    for(int c = 0; c < out_c;c++)
    {
        for(int oh = 0; oh < h; oh++)
        {
            for(int ow = 0; ow < w; ow++)
            std::cout<<buf[ow + oh * w + c * h * w]<<" ";
            std::cout<<"\n";
        }
        std::cout<<"\n";
    }
    */
    release_graph_tensor(output_tensor);
    release_graph_node(pool_node);
    postrun_graph(graph);
    destroy_graph(graph);

    free(i_buf);

    return 0;
}

int main(int argc, char* argv[])
{
    std::string plugin_file="libautokernel.so";
    if(!is_file_exist(plugin_file))
    {
        if(is_file_exist("./build/src/"+plugin_file))
        {
            plugin_file="./build/src/libautokernel.so";
        }
        else if(is_file_exist("../src/"+plugin_file))
        {
            plugin_file="../src/libautokernel.so";
        }
        else if(is_file_exist("./src/"+plugin_file))
        {
            plugin_file="./src/libautokernel.so";
        }
        else
        {
            printf("libautokernel.so not existed.\n");
        }
    }
    
    if(load_tengine_plugin("autokernel", plugin_file.c_str(), "autokernel_plugin_init")<0)
    {
        printf("init autokernel plugin failed\n");
    }
    
    printf("start init_tengine\n");
    init_tengine();
    
    //  in_c, in_h, in_w, ksize, stride, pad, pool_method
    printf("running maxpooling... \n");
    test_pool(64,    56,   60,  2, 2, 0, 0);

    printf("running avepool... \n");
    test_pool(64,    56,   60,  2, 2, 0, 1);


    release_tengine();
    return 0;
}
