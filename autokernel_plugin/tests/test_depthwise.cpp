#include <iostream>
#include <string>
#include "utils.hpp" //is_file_exist
/* the sample code to create a convolution and do calculation */

#include "tengine_c_api.h"

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

int create_input_node(graph_t graph, const char* node_name, int c, int h, int w)
{
    node_t node = create_graph_node(graph, node_name, "InputOp");
    tensor_t tensor = create_graph_tensor(graph, node_name, TENGINE_DT_FP32);
    set_node_output_tensor(node, 0, tensor, TENSOR_TYPE_INPUT);

    // int dims[4] = {1, h, w, c};	//nhwc
	int dims[4] = {1, c, h, w};  //nchw

    set_tensor_shape(tensor, dims, 4);

    release_graph_tensor(tensor);
    release_graph_node(node);

    return 0;
}

int create_conv_node(graph_t graph, const char* node_name, const char* input_name, int k_size, int stride, int pad,
                     int in_c, int out_c, int group, int activation)
{
    node_t conv_node = create_graph_node(graph, node_name, "Convolution");

    tensor_t input_tensor = get_graph_tensor(graph, input_name);

    if(input_tensor == nullptr)
    {
        std::cout << "ERRNO: " << get_tengine_errno() << "\n";
        return -1;
    }

    set_node_input_tensor(conv_node, 0, input_tensor);

    release_graph_tensor(input_tensor);

    /* output */
    tensor_t output_tensor = create_graph_tensor(graph, node_name, TENGINE_DT_FP32);
    set_node_output_tensor(conv_node, 0, output_tensor, TENSOR_TYPE_VAR);

    release_graph_tensor(output_tensor);

    /* weight */

    std::string weight_name(node_name);
    weight_name += "/weight";

    node_t w_node = create_graph_node(graph, weight_name.c_str(), "Const");
    tensor_t w_tensor = create_graph_tensor(graph, weight_name.c_str(), TENGINE_DT_FP32);
    set_node_output_tensor(w_node, 0, w_tensor, TENSOR_TYPE_CONST);
    set_node_input_tensor(conv_node, 1, w_tensor);
    // int w_dims[] = {out_c, k_size, k_size, in_c / group};	//nhwc
	int w_dims[] = {out_c, in_c / group, k_size, k_size};    //nchw

    set_tensor_shape(w_tensor, w_dims, 4);

    release_graph_node(w_node);
    release_graph_tensor(w_tensor);

    /* bias */
    std::string bias_name(node_name);
    bias_name += "/bias";

    node_t b_node = create_graph_node(graph, bias_name.c_str(), "Const");
    tensor_t b_tensor = create_graph_tensor(graph, bias_name.c_str(), TENGINE_DT_FP32);
    set_node_output_tensor(b_node, 0, b_tensor, TENSOR_TYPE_CONST);
    int b_dims[] = {out_c};

    set_tensor_shape(b_tensor, b_dims, 1);

    set_node_input_tensor(conv_node, 2, b_tensor);
    release_graph_node(b_node);
    release_graph_tensor(b_tensor);

    /* attr */
    set_node_attr_int(conv_node, "kernel_h", &k_size);
    set_node_attr_int(conv_node, "kernel_w", &k_size);
    set_node_attr_int(conv_node, "stride_h", &stride);
    set_node_attr_int(conv_node, "stride_w", &stride);
    set_node_attr_int(conv_node, "pad_h0", &pad);
    set_node_attr_int(conv_node, "pad_w0", &pad);
    set_node_attr_int(conv_node, "pad_h1", &pad);
    set_node_attr_int(conv_node, "pad_w1", &pad);
    set_node_attr_int(conv_node, "activation", &activation);
    set_node_attr_int(conv_node, "output_channel", &out_c);
    set_node_attr_int(conv_node, "group", &group);

    release_graph_node(conv_node);

    return 0;
}

graph_t create_conv_graph(int c, int h, int w, int output_c, int ksize, int stride, int pad, int group, int act)
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
    const char* conv_name = "conv";

    if(create_input_node(graph, input_name, c, h, w) < 0)
    {
        std::cerr << "create input failed\n";
        return nullptr;
    }

    if(create_conv_node(graph, conv_name, input_name, ksize, stride, pad, c, output_c, group, act) < 0)
    {
        std::cerr << "create conv node failed\n";
        return nullptr;
    }

    /* set input/output node */

    const char* inputs[] = {input_name};
    const char* outputs[] = {conv_name};

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

int test_conv(int in_c, int out_c, int h, int w, int ksize, int stride, int pad, int group, int act)
{
    graph_t graph = create_conv_graph(in_c, h, w, out_c, ksize, stride, pad, group, act);

    if(graph == nullptr)
        return 1;

    /* set input */
    tensor_t input_tensor = get_graph_input_tensor(graph, 0, 0);

    int buf_size = get_tensor_buffer_size(input_tensor);
    float* i_buf = ( float* )malloc(buf_size);
    srand(time(0));
    printf("input:\n ");
    for(unsigned int i = 0; i < buf_size / sizeof(float); i++)
    {
        i_buf[i] = i%10;//rand() % 10;
	std::cout << i_buf[i] << " ";
	if ((i + 1) % h == 0)
	    std::cout << std::endl;
    }
    std::cout << std::endl;

    set_tensor_buffer(input_tensor, i_buf, buf_size);
    release_graph_tensor(input_tensor);

    /* set weight */
    node_t conv_node = get_graph_node(graph, "conv");

    tensor_t weight_tensor = get_node_input_tensor(conv_node, 1);

    buf_size = get_tensor_buffer_size(weight_tensor);
    float* w_buf = ( float* )malloc(buf_size);

    printf("weight: \n");
    for(unsigned int i = 0; i < buf_size / sizeof(float); i++)
    {
        w_buf[i] = i%5;//rand() % 10;
    	std::cout << w_buf[i] << " ";
	if ((i + 1) % ksize == 0)
	    std::cout << std::endl;
    }
    std::cout << std::endl;

    set_tensor_buffer(weight_tensor, w_buf, buf_size);

    release_graph_tensor(weight_tensor);

    /* set bias */

    int input_num = get_node_input_number(conv_node);
    float* b_buf = nullptr;

    if(input_num > 2)
    {
        tensor_t bias_tensor = get_node_input_tensor(conv_node, 2);

        buf_size = get_tensor_buffer_size(bias_tensor);
        b_buf = ( float* )malloc(buf_size);

	printf("bias: \n");
        for(unsigned int i = 0; i < buf_size / sizeof(float); i++)
        {
            b_buf[i] = 0;
            std::cout << b_buf[i] << " ";
            if ((i + 1) % out_c == 0)
		std::cout << std::endl;
        }
        std::cout << std::endl;

        set_tensor_buffer(bias_tensor, b_buf, buf_size);
        release_graph_tensor(bias_tensor);
    }

 
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

    tensor_t output_tensor = get_node_output_tensor(conv_node, 0);

    float* buf = ( float* )get_tensor_buffer(output_tensor);
    // int size = get_tensor_buffer_size(output_tensor);
    // if(float_mismatch(buf, buf1, size/sizeof(float)) != 0)
    //     printf("test failed\n");
    std::cout<<"print output data\n";
/*    for(int i = 0; i < 14; i++)
    {
        std::cout<<buf[i]<<" ";
    }
    std::cout<<"\n";
*/
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

    release_graph_tensor(output_tensor);
    release_graph_node(conv_node);
    postrun_graph(graph);
    destroy_graph(graph);

    free(i_buf);
    free(w_buf);

    if(b_buf)
        free(b_buf);
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
    
    // in_c, out_c, in_h, out_h, k, s, p, group, act
    test_conv(2, 2, 5, 5, 3, 1, 1, 2, 0);


    release_tengine();
    return 0;
}
