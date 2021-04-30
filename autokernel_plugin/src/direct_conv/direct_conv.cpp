#include "direct_conv.h"

static int prerun(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    return 0;
}

static int run(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct node* ir_node = exec_node->ir_node;
    struct graph* ir_graph = ir_node->graph;
    struct tensor* input_tensor;
    struct tensor* weight_tensor;
    struct tensor* output_tensor;
    struct tensor* bias_tensor = NULL;
    int num_thread = exec_graph->num_thread;
    int cpu_affinity = exec_graph->cpu_affinity;

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
	int input_c = input_tensor->dims[1];
	int act = conv_param->activation;
	
	Halide::Runtime::Buffer<float> input(input_buf, input_tensor->dims[3], input_tensor->dims[2], input_tensor->dims[1], input_tensor->dims[0]);
	Halide::Runtime::Buffer<float> filter(weight_buf, weight_tensor->dims[3], weight_tensor->dims[2], weight_tensor->dims[1], weight_tensor->dims[0]);
	Halide::Runtime::Buffer<float> output(output_buf, output_tensor->dims[3], output_tensor->dims[2], output_tensor->dims[1], output_tensor->dims[0]);
	Halide::Runtime::Buffer<float> bias1(bias, output_tensor->dims[1]);
	
	halide_direct_conv(input, filter, bias1, input_c, stride, pad_width, pad_height, act, output);
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
    //printf("run halide postrun\n");
    return 0;
}

static int init_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    return 0;
}

static int release_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    return 0;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct node* exec_node)
{
    
    return 5003;
}

static struct node_ops hcl_node_ops = {.prerun = prerun,               
                                       .run = run,                   
                                       .reshape = reshape,           
                                       .postrun = postrun,              
                                       .init_node = init_node,       
                                       .release_node = release_node, 
	                               .score = score};              

int RegisterAutoKernelDirect_conv()
{
    return register_builtin_node_ops(OP_CONV, &hcl_node_ops);
}

// static int unreg_conv_hcl_ops(void* arg)
// {
//     unregister_builtin_node_ops(OP_CONV, &hcl_node_ops);
//     return 0;
// }
