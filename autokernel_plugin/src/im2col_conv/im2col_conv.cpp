#include "im2col_conv.h"

static int get_private_mem_size(struct tensor* filter)
{
    if (filter->data_type == TENGINE_DT_UINT8)    // simulator uint8 inference with fp32
        return filter->elem_num * filter->elem_size * 4;
    else
        return filter->elem_num * filter->elem_size;    // caution
}

int conv_hcl_set_shared_mem(struct conv_priv_info* priv_info, void* mem, int mem_size)
{
    priv_info->external_im2col_mem = 1;
    priv_info->im2col_buffer = mem;
    priv_info->im2col_buffer_size = mem_size;
    return 0;
}

int conv_hcl_get_shared_mem_size(struct tensor* input, struct tensor* output, struct conv_param* param)
{
    int group = param->group;
    int input_chan = param->input_channel / group;
    int kernel_size = input_chan * param->kernel_h * param->kernel_w;
    int output_xy = output->dims[2] * output->dims[3];
    int elem_size = input->elem_size;

    // simulator uint8 inference with fp32
    if (input->data_type == TENGINE_DT_UINT8)
        elem_size = 4;
    return elem_size * output_xy * kernel_size;
}

int conv_hcl_get_shared_pack4_mem_size(struct tensor* filter, struct tensor* output, struct conv_param* param)
{
    int K = filter->elem_num / filter->dims[0];
    int N = output->dims[2] * output->dims[3];
    int elem_size = filter->elem_size;

    // simulator uint8 inference with fp32
    if (filter->data_type == TENGINE_DT_UINT8)
        elem_size = 4;

    return (8 * K * (N / 8 + N % 8)) * elem_size;
}

static void interleave(struct tensor* filter, struct conv_priv_info* priv_info)
{
    /* simply copy the data */
    memcpy(priv_info->interleave_buffer, filter->data, filter->elem_num * filter->elem_size);
}

void im2col(float* data_img, float* data_col, int inh, int inw, int inc, int outh, int outw, int outc, int ksize_h, int ksize_w, int sh, int sw, int ph, int pw, int dh, int dw)
{
    const int channels_col = ksize_h * ksize_w * inc;

    for(int c = 0; c < channels_col; ++c)
    {
        const int kw = c % ksize_w;
        int c_ = c / ksize_w;
        const int kh = c_ % ksize_h;
        c_ = c_ / ksize_h;
        const int im_col = kw * dw - pw;
        const int w_low = std::max(0, -im_col / sw + (-im_col % sw > 0));
        const int w_high = std::min(outw, (inw - im_col) / sw + ((inw - im_col) % sw > 0));
        for(int h = 0; h < outh; ++h)
	{
	    const int im_row = kh * dh + h * sh - ph;
	    float* out = data_col + (c * outh + h) * outw;
	    const float* end = out + w_high;

	    if(im_row >= 0 && im_row < inh)
	    {
                float* in = data_img + inw * (im_row + inh * c_) + im_col + (w_low - 1) * sw;
	        memset(out, 0, w_low * sizeof(float));
		out += w_low;
		while(out < end)
		{
		    in += sw;
	            *(out++) = *in;
		}
		memset(out, 0, (outw - w_high) * sizeof(float));
	    }
		else
	    {
     		memset(out, 0, outw * sizeof(float));
            }
	}
    }
}

void add_bias(float* output, float* bias, int c_out, int hw)
{
    for(int c = 0; c < c_out; ++c)
    {
        for(int i = 0; i < hw; ++i)
        {
            output[c * hw + i] += bias[c];
        }
    }
}

void relu(float* data, int size, int activation)
{
    for(int i = 0; i < size; i++)
    {
        data[i] = std::max(data[i], ( float )0);
        if(activation > 0)
        {
            data[i] = std::min(data[i], ( float )activation);
        }
    }
}

static int prerun(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct node* ir_node = exec_node->ir_node;
    struct graph* ir_graph = ir_node->graph;
    struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct tensor* filter_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);
    struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    struct conv_param* conv_param = ( struct conv_param* )ir_node->op.param_mem;
    struct conv_priv_info* conv_priv_info = ( struct conv_priv_info* )exec_node->ops_priv;

    /* get cpu affinity */
    conv_priv_info->cpu_type = exec_graph->cpu_affinity;
    
    if (exec_graph->mode == TENGINE_MODE_FP32 || exec_graph->mode == TENGINE_MODE_UINT8)
    {
        if (exec_node->shared_mem_size < exec_graph->shared_mem_size)
	{
            if (conv_hcl_set_shared_mem(conv_priv_info, exec_graph->shared_mem, exec_graph->shared_mem_size) < 0)
	    {
                printf("halide im2col+gemm: set shared memory failed\n");
		// set_tengine_errno(EFAULT);
		return -1;
	    }
	}
	
	conv_priv_info->external_interleave_pack4_mem = 0;
        
	/* do prerun interleave */
	if (!conv_priv_info->external_im2col_mem)
	{
	    int mem_size = conv_hcl_get_shared_mem_size(input_tensor, output_tensor, conv_param);
	    void* mem = sys_malloc(mem_size);
	    conv_priv_info->im2col_buffer = mem;
	    conv_priv_info->im2col_buffer_size = mem_size;
	}

	if (!conv_priv_info->external_interleave_mem)
	{
	    int mem_size = get_private_mem_size(filter_tensor);
	    void* mem = sys_malloc(mem_size);
	    conv_priv_info->interleave_buffer = mem;
	    conv_priv_info->interleave_buffer_size = mem_size;
	}

	if (input_tensor->data_type == TENGINE_DT_UINT8)
	{
	    printf("not support uint8 for now, fix me\n");
	}
	else
	{
	    interleave(filter_tensor, conv_priv_info);
	}


    }
    else
    {
        printf("Tengine work node not support %d\n", exec_graph->mode);
	return -1;
    }

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
    struct node* ir_node = exec_node->ir_node;
    struct graph* ir_graph = ir_node->graph;
    struct tensor* input_tensor;
    struct tensor* weight_tensor;
    struct tensor* output_tensor;
    struct tensor* bias_tensor = NULL;
    // int num_thread = exec_graph->num_thread;
    // int cpu_affinity = exec_graph->cpu_affinity;

    /* set the input data and shape again, in case of reshape or dynamic shape */
    input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    weight_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);
    output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);
    if (ir_node->input_num > 2)
        bias_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[2]);

    struct conv_param* conv_param = ( struct conv_param* )ir_node->op.param_mem;
    struct conv_priv_info* conv_priv_info = ( struct conv_priv_info* )exec_node->ops_priv;

    /* fp32 run */
    int group = conv_param->group;
    int inc_g = (input_tensor->dims[1])/group;
    int outc_g = (output_tensor->dims[1])/group;
    int ksize_h = conv_param->kernel_h;
    int ksize_w = conv_param->kernel_w;
    int stride_h = conv_param->stride_h;
    int stride_w = conv_param->stride_w;
    int pad_h = conv_param->pad_h0;
    int pad_w = conv_param->pad_w0;
    int dilation_h = conv_param->dilation_h;
    int dilation_w = conv_param->dilation_w;
    int activation = conv_param->activation;

    if (exec_graph->mode == TENGINE_MODE_FP32)
    {
        int K = weight_tensor->elem_num / weight_tensor->dims[0];
	int N = output_tensor->dims[2] * output_tensor->dims[3];
	int M = output_tensor->dims[1] / group;                   

        for (int i = 0; i < input_tensor->dims[0]; i++)
	{
            for(int j = 0; j < group; j++)
	    {
		im2col((float*)(input_tensor->data), (float*)(conv_priv_info->im2col_buffer), input_tensor->dims[2], input_tensor->dims[3], inc_g, output_tensor->dims[2], output_tensor->dims[3], outc_g, ksize_h, ksize_w, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w);
		
		{
		    Halide::Runtime::Buffer<float> filter_data((float*)conv_priv_info->interleave_buffer, K, M);
		    Halide::Runtime::Buffer<float> output_data((float*)output_tensor->data, N, M);
		    Halide::Runtime::Buffer<float> input_data((float*)conv_priv_info->im2col_buffer, N, K);
		    halide_im2col_conv(filter_data, input_data, output_data);
		}
	    }
	}
	if(ir_node->input_num > 2)
	{
	    float* bias_data = (float*)bias_tensor->data;
	    for(int i = 0; i < output_tensor->dims[0]; i++)
	    {
		add_bias((float*)output_tensor->data, bias_data, output_tensor->dims[1], output_tensor->dims[2]*output_tensor->dims[3]);
	    }
	}
    if(info_autokernel)printf("[INFO]: runing AutoKernel im2col_conv ...\n");
	if(activation >= 0)
	{
	    relu((float*)output_tensor->data, output_tensor->elem_num, activation);
	}
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
    struct conv_priv_info* conv_priv_info = ( struct conv_priv_info* )exec_node->ops_priv;

    /* fp32 postrun */
    if (exec_graph->mode == TENGINE_MODE_FP32 || exec_graph->mode == TENGINE_MODE_UINT8)
    {
        if (conv_priv_info->external_interleave_pack4_mem && !conv_priv_info->external_interleave_mem && conv_priv_info->interleave_buffer != NULL)
        {
            sys_free(conv_priv_info->interleave_buffer_pack4);
            conv_priv_info->interleave_buffer_pack4 = NULL;
        }

        if (!conv_priv_info->external_im2col_mem && conv_priv_info->im2col_buffer != NULL)
        {
            sys_free(conv_priv_info->im2col_buffer);
            conv_priv_info->im2col_buffer = NULL;
        }
        if (!conv_priv_info->external_im2col_pack4_mem && conv_priv_info->im2col_buffer_pack4 != NULL)
        {
            sys_free(conv_priv_info->im2col_buffer_pack4);
            conv_priv_info->im2col_buffer_pack4 = NULL;
        }
        if (conv_priv_info->external_interleave_pack4_mem && conv_priv_info->interleave_buffer_pack4 != NULL)
        {
            sys_free(conv_priv_info->interleave_buffer_pack4);
            conv_priv_info->interleave_buffer_pack4 = NULL;
        }
    }
    else
    {
        printf("Tengine work node not support %d\n", exec_graph->mode);
	return -1;
    }
    return 0;
}

static int init_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct node* ir_node = exec_node->ir_node;
    struct graph* ir_graph = ir_node->graph;
    struct tensor* input_tensor;
    struct tensor* filter_tensor;
    struct tensor* output_tensor;

    input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    filter_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);
    output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    struct conv_param* conv_param = ( struct conv_param* )ir_node->op.param_mem;

    /* init the private info data of convolution op */
    struct conv_priv_info* conv_priv_info = ( struct conv_priv_info* )sys_malloc(sizeof(struct conv_priv_info));
    if (conv_priv_info == NULL)
    {
        // set_tengine_errno(ENOMEM);
	return -1;
    }
    memset(conv_priv_info, 0, sizeof(struct conv_priv_info));
    exec_node->ops_priv = conv_priv_info;

    /* get shared memory size */
    if (exec_graph->mode == TENGINE_MODE_FP32 || exec_graph->mode == TENGINE_MODE_UINT8)
    {
        exec_node->shared_mem_size = conv_hcl_get_shared_mem_size(input_tensor, output_tensor, conv_param);
        exec_node->shared_pack4_mem_size = conv_hcl_get_shared_pack4_mem_size(filter_tensor, output_tensor, conv_param);
    }
    else
    {
        printf("Tengine work node not support %d\n", exec_graph->mode);
	return -1;
    }
    return 0;
}

static int release_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct conv_priv_info* conv_priv_info = ( struct conv_priv_info* )exec_node->ops_priv;
    sys_free(conv_priv_info);
    exec_node->ops_priv = NULL;
    return 0;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct node* exec_node)
{
    
    return 6002;
}

static struct node_ops hcl_node_ops = {.prerun = prerun,               
                                       .run = run,                   
                                       .reshape = reshape,           
                                       .postrun = postrun,              
                                       .init_node = init_node,       
                                       .release_node = release_node, 
	                               .score = score};              

int RegisterAutoKernelIm2col_conv()
{
    return register_builtin_node_ops(OP_CONV, &hcl_node_ops);
}

// static int unreg_conv_hcl_ops(void* arg)
// {
//     unregister_builtin_node_ops(OP_CONV, &hcl_node_ops);
//     return 0;
// }


