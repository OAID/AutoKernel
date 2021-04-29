#include "fc.h"

// add helper data struct and functions here
/*
struct op_priv_info
{

};
*/

static int prerun(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    /*
    allocate helper memory for your op
    */
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
    // step 1: get input and output
    struct node* ir_node = exec_node->ir_node;
    struct graph* ir_graph = ir_node->graph;
    struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct tensor* weight_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);
    struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);
    struct tensor* bias_tensor = NULL;
    if (ir_node->input_num > 2)
	bias_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[2]);
    struct fc_data* fc_param = ( struct fc_data* )ir_node->op.param_mem;

    float* input_buf = (float*)(input_tensor->data);
    float* weight_buf = (float*)(weight_tensor->data);
    float* output_buf = (float*)(output_tensor->data);
    float* bias = NULL;
    if(ir_node->input_num > 2)
	bias = (float*)(bias_tensor->data);

    if(exec_graph->mode == TENGINE_MODE_FP32)
    {
        Halide::Runtime::Buffer<float> input(input_buf, input_tensor->dims[1], input_tensor->dims[0]);
        Halide::Runtime::Buffer<float> weight(weight_buf, weight_tensor->dims[1], weight_tensor->dims[0]);  
        Halide::Runtime::Buffer<float> output(output_buf, output_tensor->dims[1], output_tensor->dims[0]);
	Halide::Runtime::Buffer<float> bias1(bias, output_tensor->dims[1]);

        printf("[INFO]:using halide fc...\n");

        halide_fc(input, weight, bias1, output);
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
    return 0;
}

static int init_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
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

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct node* exec_node)
{
    /*
    OPS_SCORE_STATIC 10000
    OPS_SCORE_BEST 8000
    OPS_SCORE_PREFER 6000
    OPS_SCORE_CANDO 4000
    OPS_SCORE_NOTSUP 2000
    */
    return OPS_SCORE_STATIC;
}

static struct node_ops autokernel_node_ops = {.prerun = prerun,
                                       .run = run,
                                       .reshape = reshape,
                                       .postrun = postrun,
                                       .init_node = init_node,
                                       .release_node = release_node,
                                       .score = score};

int RegisterAutoKernelFc()
{
    return register_builtin_node_ops(OP_FC, &autokernel_node_ops);
}

//static int unreg_autokernel_ops(void* arg)
//{
//    unregister_builtin_node_ops(OP_FC, &autokernel_node_ops);
//    return 0;
//}
