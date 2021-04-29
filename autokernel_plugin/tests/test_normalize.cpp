#include <iostream>
#include <string>
#include "utils.hpp" //is_file_exist
/* the sample code to create a convolution and do calculation */

#include "tengine/c_api.h"

static inline unsigned long get_cur_time(void)
{
    struct timespec tm;

    clock_gettime(CLOCK_MONOTONIC, &tm);

    return (tm.tv_sec * 1000000 + tm.tv_nsec / 1000);
}

int float_mismatch(float* a, float* b, int size)
{
    int i =0;

    for(i=0;i<size;i++)
    {
        float off = a[i] - b[i];
        if(off!=0)
        {
            std::cout <<"mismatch:\t["<<i<<"]\ta:"<<a[i] <<"\tb:"<<b[i]<<"\toff:"<<a[i]-b[i]<<"\n";
            break;
        }
    }
    if(i!= size)
    {
        printf("mismatch:\n\t[%d]\t---a:    %f ,%f   :b---        off: %f\n",i,a[i],b[i],a[i]-b[i]);
        printf("fail\n");
        return -1;
    }
    printf("pass\n");
    return 0;
}

int create_input_node(graph_t graph, const char* node_name, int n, int c, int h, int w)
{
    node_t node = create_graph_node(graph, node_name, "InputOp");
    tensor_t tensor = create_graph_tensor(graph, node_name, TENGINE_DT_FP32);
    set_node_output_tensor(node, 0, tensor, TENSOR_TYPE_INPUT);

    int dims[4] = {n, c, h, w};  //nchw

    set_tensor_shape(tensor, dims, 4);

    release_graph_tensor(tensor);
    release_graph_node(node);

    return 0;
}

int create_normalize_node(graph_t graph, const char* node_name, const char* input_name, int n, int c, int h, int w)
{
    node_t normalize_node = create_graph_node(graph, node_name, "Normalize");

    tensor_t input_tensor = get_graph_tensor(graph, input_name);

    if(input_tensor == nullptr)
    {
        std::cout << "ERRNO: " << get_tengine_errno() << "\n";
        return -1;
    }
    set_node_input_tensor(normalize_node, 0, input_tensor);
    release_graph_tensor(input_tensor);

    /* output */ tensor_t output_tensor = create_graph_tensor(graph, node_name, TENGINE_DT_FP32);
    set_node_output_tensor(normalize_node, 0, output_tensor, TENSOR_TYPE_VAR);

    release_graph_tensor(output_tensor);

    /* scale */
    std::string weight_name(node_name);
    weight_name += "/scale";

    node_t scale_node = create_graph_node(graph, weight_name.c_str(), "Const");
    tensor_t scale_tensor = create_graph_tensor(graph, weight_name.c_str(), TENGINE_DT_FP32);
    set_node_output_tensor(scale_node, 0, scale_tensor, TENSOR_TYPE_CONST);
    set_node_input_tensor(normalize_node, 1, scale_tensor);

    int scale_dims[] = {c};
    set_tensor_shape(scale_tensor, scale_dims, 1);

    release_graph_node(scale_node);
    release_graph_tensor(scale_tensor);

    release_graph_node(normalize_node);

    return 0;
}

graph_t create_normalize_graph(int n, int c, int h, int w)
{
    graph_t graph = create_graph(nullptr, nullptr, nullptr);
    set_graph_layout(graph, TENGINE_LAYOUT_NCHW);    //nchw

    if(graph == nullptr)
    {
        std::cerr << "ERRNO: " << get_tengine_errno() << "\n";
        return nullptr;
    }

    const char* input_name = "data";
    const char* normalize_name = "normalize";

    if(create_input_node(graph, input_name, n, c, h, w) < 0)
    {
        std::cerr << "create input failed\n";
        return nullptr;
    }

    if(create_normalize_node(graph, normalize_name, input_name, n, c, h, w) < 0)
    {
        std::cerr << "create normalize node failed\n";
        return nullptr;
    }

    /* set input/output node */

    const char* inputs[] = {input_name};
    const char* outputs[] = {normalize_name};

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

int test_normalize(int n, int c, int h, int w)
{
    graph_t graph = create_normalize_graph(n, c, h, w);
    
    if(graph == nullptr)
        return 1;

    /* set input */
    tensor_t input_tensor = get_graph_input_tensor(graph, 0, 0);

    int buf_size = get_tensor_buffer_size(input_tensor);
    float* i_buf = ( float* )malloc(buf_size);
    srand(time(0));
    for(unsigned int i = 0; i < buf_size / sizeof(float); i++)
    {
        i_buf[i] = i%10;//rand() % 10;
    }

    set_tensor_buffer(input_tensor, i_buf, buf_size);
    release_graph_tensor(input_tensor);

    /* set scale */
    node_t normalize_node = get_graph_node(graph, "normalize");

    tensor_t scale_tensor = get_node_input_tensor(normalize_node, 1);

    buf_size = get_tensor_buffer_size(scale_tensor);
    float* s_buf = ( float* )malloc(buf_size);

    for(unsigned int i = 0; i < buf_size / sizeof(float); i++)
    {
        s_buf[i] = i%5;//rand() % 10;
    }

    set_tensor_buffer(scale_tensor, s_buf, buf_size);

    release_graph_tensor(scale_tensor);
 
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

    tensor_t output_tensor = get_node_output_tensor(normalize_node, 0);

    float* buf = ( float* )get_tensor_buffer(output_tensor);
    // int size = get_tensor_buffer_size(output_tensor);
    // if(float_mismatch(buf, buf1, size/sizeof(float)) != 0)
    //     printf("test failed\n");
    std::cout<<"print output data\n";
    for(int i = 0; i < (n * c * h * w); i++)
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
    release_graph_node(normalize_node);
    postrun_graph(graph);
    destroy_graph(graph);

    free(i_buf);
    free(s_buf);

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
    printf("init_tengine done\n");

    int n, c, h, w;
    n = 2;
    c = 3;
    h = w = 2;

    test_normalize(n, c, h, w);

    release_tengine();
    return 0;
}
