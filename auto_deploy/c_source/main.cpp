#include<stdio.h>
#include<stdlib.h>
#include "halide_conv.h"
#include "halide_relu.h"
#include "halide_maxpool.h"
#include "halide_matmul.h"
#include "HalideRuntime.h"

typedef struct Param {
    int* inp0_dims;
    int* inp1_dims;
    int* inp2_dims;
    int* inp3_dims;
    int* out0_dims;

    int ksize;
    int stride;
    int pad;
}Param;

void read_float_data(float* data, int size, char* fname)
{
    FILE* fp = fopen(fname, "rb");
    if (!fp) printf("data can not be open");
    fread(data, sizeof(float), size, fp);
    fclose(fp);
}
void p(float* data,int size)
{
    for(int i=0;i<size;i++)
    printf("%f ",data[i]);
    printf("\n");
}

void set_data(struct halide_buffer_t* buf,int* shape,int n,void* data)
{
    //dims
    buf->dimensions=n;
    int step[4]={1,1,1,1};
    for(int i=1;i<n;i++)
    {
        step[i]=step[i-1]*shape[i-1];
    }
    buf->dim=(halide_dimension_t*)malloc(sizeof(halide_dimension_t)*n);

    for(int i=0;i<n;i++)
    {
        buf->dim[i].min=0;
        buf->dim[i].extent=shape[i];
        buf->dim[i].stride=step[i];
    }
 
    //type
    buf->type.bits=32;
    buf->type.lanes=1;
    buf->type.code=halide_type_float;


    buf->host=(uint8_t*)data;
    buf->flags=0;
    buf->device=0;
}
void Conv_Add_fused(float* out0, float* inp0, float* inp1, float* inp2, Param* param){
    struct halide_buffer_t b_out0;
    struct halide_buffer_t b_inp0, b_inp1, b_inp2;
    set_data(&b_out0,param->out0_dims,4,out0);
    set_data(&b_inp0,param->inp0_dims,4,inp0);
    set_data(&b_inp1,param->inp1_dims,4,inp1);
    set_data(&b_inp2,param->inp2_dims,1,inp2);
    halide_conv(&b_inp0, &b_inp1, &b_inp2, param->stride, param->pad, &b_out0);
    free(b_out0.dim);
    free(b_inp0.dim);
    free(b_inp1.dim);
    free(b_inp2.dim);
}
void Relu(float* out0, float* inp0, Param* param){
    struct halide_buffer_t b_out0;
    struct halide_buffer_t b_inp0;
    set_data(&b_out0,param->out0_dims,4,out0);
    set_data(&b_inp0,param->inp0_dims,4,inp0);
    halide_relu(&b_inp0, &b_out0);
    free(b_out0.dim);
    free(b_inp0.dim);
}
void MaxPool(float* out0, float* inp0, Param* param){
    struct halide_buffer_t b_out0;
    struct halide_buffer_t b_inp0;
    set_data(&b_out0,param->out0_dims,4,out0);
    set_data(&b_inp0,param->inp0_dims,4,inp0);
    halide_maxpool(&b_inp0, param->ksize, param->stride, &b_out0);
    free(b_out0.dim);
    free(b_inp0.dim);
}
void MatMul_Add_fused(float* out0, float* inp0, float* inp1, float* inp2, Param* param){
    struct halide_buffer_t b_out0;
    struct halide_buffer_t b_inp0, b_inp1, b_inp2;
    set_data(&b_out0,param->out0_dims,2,out0);
    set_data(&b_inp0,param->inp0_dims,2,inp0);
    set_data(&b_inp1,param->inp1_dims,2,inp1);
    set_data(&b_inp2,param->inp2_dims,1,inp2);
    halide_matmul(&b_inp0, &b_inp1, &b_inp2, &b_out0);
    free(b_out0.dim);
    free(b_inp0.dim);
    free(b_inp1.dim);
    free(b_inp2.dim);
}
int main(int argc, char** argv){
    if(argc<3){printf("exe <model> <inp_data>\n");return 0;}
    char* weight_name=argv[1];
    char* input_data_file=argv[2];

    //data
    float* _0= (float*)malloc(sizeof(float)*784); //Input3
    float* _1= (float*)malloc(sizeof(float)*200); //Parameter5
    float* _2= (float*)malloc(sizeof(float)*8); //Parameter6
    float* _3= (float*)malloc(sizeof(float)*6272); //Plus30_Output_0
    float* _4= (float*)malloc(sizeof(float)*6272); //ReLU32_Output_0
    float* _5= (float*)malloc(sizeof(float)*1568); //Pooling66_Output_0
    float* _6= (float*)malloc(sizeof(float)*3200); //Parameter87
    float* _7= (float*)malloc(sizeof(float)*16); //Parameter88
    float* _8= (float*)malloc(sizeof(float)*3136); //Plus112_Output_0
    float* _9= (float*)malloc(sizeof(float)*3136); //ReLU114_Output_0
    float* _10= (float*)malloc(sizeof(float)*256); //Pooling160_Output_0
    float* _11= (float*)malloc(sizeof(float)*2560); //Parameter193_reshape1
    float* _12= (float*)malloc(sizeof(float)*10); //Parameter194
    float* _13= (float*)malloc(sizeof(float)*10); //Plus214_Output_0

    //load_weight
    FILE* fp = fopen(weight_name, "rb");
    if (!fp) printf("data can not be open");
    fread(_1, sizeof(float), 200, fp);
    fread(_2, sizeof(float), 8, fp);
    fread(_6, sizeof(float), 3200, fp);
    fread(_7, sizeof(float), 16, fp);
    fread(_11, sizeof(float), 2560, fp);
    fread(_12, sizeof(float), 10, fp);
    fclose(fp);

    //read input data
    read_float_data(_0,784,input_data_file);

    //data shape
    int s_0[4]={28,28,1,1};
    int s_1[4]={5,5,1,8};
    int s_2[1]={8};
    int s_3[4]={28,28,8,1};
    int s_4[4]={28,28,8,1};
    int s_5[4]={14,14,8,1};
    int s_6[4]={5,5,8,16};
    int s_7[1]={16};
    int s_8[4]={14,14,16,1};
    int s_9[4]={14,14,16,1};
    int s_10[2]={256,1};
    int s_11[2]={10,256};
    int s_12[1]={10};
    int s_13[2]={10,1};

    //param
    Param param_0;
    Param param_1;
    Param param_2;
    Param param_3;
    Param param_4;
    Param param_5;
    Param param_6;
    param_0.inp0_dims=s_0;
    param_0.inp1_dims=s_1;
    param_0.inp2_dims=s_2;
    param_0.out0_dims=s_3;
    param_0.stride=1;param_0.pad=2;//conv
    param_1.inp0_dims=s_3;
    param_1.out0_dims=s_4;
    param_2.inp0_dims=s_4;
    param_2.out0_dims=s_5;
    param_2.ksize=2;param_2.stride=2;//maxpool
    param_3.inp0_dims=s_5;
    param_3.inp1_dims=s_6;
    param_3.inp2_dims=s_7;
    param_3.out0_dims=s_8;
    param_3.stride=1;param_3.pad=2;//conv
    param_4.inp0_dims=s_8;
    param_4.out0_dims=s_9;
    param_5.inp0_dims=s_9;
    int reshape_10[4]={4,4,16,1};
    param_5.out0_dims=reshape_10;
    param_5.ksize=3;param_5.stride=3;//maxpool
    param_6.inp0_dims=s_10;
    param_6.inp1_dims=s_11;
    param_6.inp2_dims=s_12;
    param_6.out0_dims=s_13;

    //code_inference
    Conv_Add_fused(_3,_0,_1,_2,&param_0);
    Relu(_4,_3,&param_1);
    MaxPool(_5,_4,&param_2);
    Conv_Add_fused(_8,_5,_6,_7,&param_3);
    Relu(_9,_8,&param_4);
    MaxPool(_10,_9,&param_5);
    MatMul_Add_fused(_13,_10,_11,_12,&param_6);

    //print output data[:10]
    p(_13,10);

    //free data
    free(_0);
    free(_1);
    free(_2);
    free(_3);
    free(_4);
    free(_5);
    free(_6);
    free(_7);
    free(_8);
    free(_9);
    free(_10);
    free(_11);
    free(_12);
    free(_13);
    return 0;
}
