#include "depthwise.h"

// add helper data struct and functions here

static int prerun(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    return 0;
}

static int run(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    bool info_autokernel = false;
    const char* debug_env = std::getenv("DEBUG_INFO");
    if((debug_env) && (debug_env[0] == '1'))
    {
        info_autokernel = true;
    }
    struct ir_node* ir_node = exec_node->ir_node;
    struct ir_graph* ir_graph = ir_node->graph;
      struct ir_tensor* input_tensor;
    struct ir_tensor* weight_tensor;
    struct ir_tensor* output_tensor;
    struct ir_tensor* bias_tensor = NULL;
    // int num_thread = exec_graph->num_thread;
    // int cpu_affinity = exec_graph->cpu_affinity;

    /* set the input data and shape again, in case of reshape or dynamic shape */
    input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    weight_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);
    output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);
    if (ir_node->input_num > 2)
        bias_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[2]);

    struct conv_param* conv_param = ( struct conv_param* )ir_node->op.param_mem;

    float* input_buf = (float*)(input_tensor->data);
    float* weight_buf = (float*)(weight_tensor->data);
    float* output_buf = (float*)(output_tensor->data);
    float* bias = NULL;
    if (ir_node->input_num > 2)
        bias = (float*)(bias_tensor->data);

    if (exec_graph->mode == TENGINE_MODE_FP32)
    {
        int stride = conv_param->stride_h;
        int pad_width = conv_param->pad_w0;
        int pad_height = conv_param->pad_h0;
	int act = conv_param->activation;
        int group = conv_param->group;

        Halide::Runtime::Buffer<float> input(input_buf, input_tensor->dims[3], input_tensor->dims[2], input_tensor->dims[1], input_tensor->dims[0]);
        Halide::Runtime::Buffer<float> filter(weight_buf, weight_tensor->dims[3], weight_tensor->dims[2], weight_tensor->dims[1], weight_tensor->dims[0]);
        Halide::Runtime::Buffer<float> output(output_buf, output_tensor->dims[3], output_tensor->dims[2], output_tensor->dims[1], output_tensor->dims[0]);
        Halide::Runtime::Buffer<float> bias1(bias, output_tensor->dims[1]);

        if(info_autokernel)printf("[INFO]: runing AutoKernel im2col_conv ...\n");

        halide_depthwise(input, filter, bias1, stride, pad_width, pad_height, act, output);
    }
    else
    {
        printf("Tengine work node not support %d\n", exec_graph->mode);
	return -1;
    }
    return 0;
}

static int reshape(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    return 0;
}

static int postrun(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    /*
    release the helper memory you 
    */
    return 0;
}

static int init_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    /* 
    init the private info data for your op:
    void ops_priv;
    int shared_mem_size;
    int shared_pack4_mem_size;
    */
    return 0;
}

static int release_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    /* 
    release the private info data for your op:
    void ops_priv;
    int shared_mem_size;
    int shared_pack4_mem_size;
    */
    return 0;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct ir_node* exec_node)
{
    struct conv_param* param = ( struct conv_param* )exec_node->op.param_mem;
    struct ir_node* ir_node = exec_node;
    struct ir_graph* ir_graph = ir_node->graph;

    struct ir_tensor* input_tensor;
    struct ir_tensor* output_tensor;

    int group = param->group;
    int kernel_h = param->kernel_h;
    int kernel_w = param->kernel_w;
    int stride_h = param->stride_h;
    int stride_w = param->stride_w;
    int dilation_h = param->dilation_h;
    int dilation_w = param->dilation_w;
    int pad_h0 = param->pad_h0;
    int pad_w0 = param->pad_h0;
    int pad_h1 = param->pad_h1;
    int pad_w1 = param->pad_w1;

    input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);
  
    int in_c = input_tensor->dims[1] / group;
    int out_c = output_tensor->dims[1] / group;

    if (input_tensor->data_type != TENGINE_DT_FP32)
	return 0;
    if (kernel_h != kernel_w || input_tensor->dims[0] > 1)
	return 0;
    
    if (param->group > 1 && in_c == 1 && out_c == 1 && pad_h0 == pad_h1 && pad_w0 == pad_w1 
       && dilation_h == 1 && dilation_w == 1 && kernel_h == 3 && kernel_w == 3
       && ((stride_h == 1 && stride_w == 1) || (stride_w == 2 && stride_h == 2)))
    {    
        return OPS_SCORE_STATIC;
    }
    else
    {
	return 0;
    }
}

static struct node_ops autokernel_node_ops = {.prerun = prerun,
                                       .run = run,
                                       .reshape = reshape,
                                       .postrun = postrun,
                                       .init_node = init_node,
                                       .release_node = release_node,
                                       .score = score};

static int reg_autokernel_ops(void* arg)
{
    return register_builtin_node_ops(OP_CONV, &autokernel_node_ops);
}

//static int unreg_autokernel_ops(void* arg)
//{
//    unregister_builtin_node_ops(OP_DEPTHWISE, &autokernel_node_ops);
//    return 0;
//}

void RegisterAutoKernelDepthwise()
{
    register_norm_module_init(2, "reg_autokernel_ops", reg_autokernel_ops, NULL);
}

