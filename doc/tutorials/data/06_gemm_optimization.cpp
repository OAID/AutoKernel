#include "Halide.h"
#include <iostream>
#include<cblas.h>
#include <cmath>
#include <sys/time.h>
using namespace Halide;
unsigned long get_cur_time(void)
{
    struct timeval tv;

    gettimeofday(&tv, NULL);

    return (tv.tv_sec * 1000000 + tv.tv_usec);
}
#ifndef MAX
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#endif
static inline float *init(int size, int mode)
{
    srand(0); //set rand_seed
    int i;
    float *m = (float *)malloc(size * sizeof(float));
    for (i = 0; i < size; ++i) {
        if (mode == 0)
            m[i] = 0;
        else if (mode == 1)
            m[i] = 1;
        else if (mode == 2)
            m[i] = i % 8;
        else if (mode == 3)
            m[i] = (float)(rand()%4); 
        else
            m[i] = (float)rand() / RAND_MAX;
    }
    return m;
}
void maxerr(float* pred, float* gt, int h,int w)
{
    float maxError = 0.f;

    for(int i=0; i< (h*w); i++){
            maxError = MAX(( float )fabs(gt[i] - pred[i]), maxError);
    }
    // printf("====================================\n");
    printf("err %.2f\t", maxError);
    // printf("====================================\n");
}

int main(int argc, char **argv) {
    if(argc<2)
    {
        printf("[usage] exe [step] <rep=30> <debug=0>\n");
        return 1;
    } 
    int M= 640;
    int N= 640;
    int K= 640;
    printf("M N K = %3d %3d %3d\t",M,N,K);
    int debug=0;
    int repeat_count=50;

    int step = atoi(argv[1]);

    float* a = init(M*K,4);
    float* b = init(N*K,4);
    float* c = init(M*N,1);
    float* ct = init(M*N,2);

    Buffer<float> A(a,K,M);
    Buffer<float> B(b,N,K);
    Buffer<float> C(c,N,M);
    
    Var x,y,xy;
    Var xi,yi,xo,yo,yii;
    RDom k(0, K);
    Func gemm("gemm");

    //1: default
    if (step==1)
    {
        gemm(x, y) += A(k, y) * B(x, k);

    }
    //2: tile
    if(step==2)
    {
        gemm(x, y) += A(k, y) * B(x, k);
        gemm.update()
            .tile(x, y, xo, yo, xi, yi, 16, 8)
            .reorder(xi, yi, k, xo, yo);
    }
    //3 tile + vectorize
    if(step==3)
    {
        gemm(x, y) += A(k, y) * B(x, k);
        gemm.update()
            .tile(x, y, xo, yo, xi, yi, 16, 8)
            .reorder(xi, yi, k, xo, yo)
            .vectorize(xi, 8);
    }
    //4 tile + vectorize + parallel
    if(step==4)
    {
        gemm(x, y) += A(k, y) * B(x, k);
        gemm.update()
            .tile(x, y, xo, yo, xi, yi, 16, 8)
            .reorder(xi, yi, k, xo, yo)
            .vectorize(xi, 8)
            .parallel(yo);
    }
    //5 tile + vectorize + parallel + unroll
    if(step==5)
    {
        gemm(x, y) += A(k, y) * B(x, k);
        gemm.update()
            .tile(x, y, xo, yo, xi, yi, 16, 8)
            .reorder(xi, yi, k, xo, yo)
            .vectorize(xi, 8)
            .parallel(yo)
            .unroll(xi)
            .unroll(yi,2);
    } 
    //6 micro_kernel 4x16
    if(step==6)
    {
        Func prod;
        prod(x, y) += A(k, y) * B(x, k);
        gemm(x, y) = prod(x, y);

        gemm.tile(x, y, xi, yi, 16, 32)
            .fuse(x, y, xy).parallel(xy)
            .split(yi, yi, yii, 4)
            .vectorize(xi, 8)
            .unroll(xi)
            .unroll(yii);

        prod.compute_at(gemm, yi)
            .vectorize(x, 8).unroll(y);

        prod.update()
            .reorder(x, y, k)
            .vectorize(x, 8)
            .unroll(x)
            .unroll(y)
            .unroll(k, 2);
    }
    // 7.interleave B
    if(step==7)
    {
        Func B_interleave("B"), Bs("Bs");
        Bs(x, y, xo) = B(xo * 16 + x, y);
        B_interleave(x, y) = Bs(x % 16, y, x / 16);

        Func prod;
        prod(x, y) += A(k, y) * B_interleave(x, k);
        gemm(x, y) = prod(x, y);

        gemm.tile(x, y, xi, yi, 16, 32)
            .fuse(x, y, xy).parallel(xy)
            .split(yi, yi, yii, 4)
            .vectorize(xi, 8)
            .unroll(xi)
            .unroll(yii);

        prod.compute_at(gemm, yi)
            .vectorize(x, 8).unroll(y);

        prod.update()
            .reorder(x, y, k)
            .vectorize(x, 8)
            .unroll(x)
            .unroll(y)
            .unroll(k, 2);
        Bs.compute_root()
            .split(y, yo, yi, 16)
            .reorder(x, yi, xo, yo)
            .unroll(x)
            .vectorize(yi).parallel(yo, 4);
    }
    gemm.output_buffer().dim(0).set_bounds(0, N).dim(1).set_bounds(0, M);
    gemm.realize(C);

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1, a, K, b, N, 0, ct, N);
    maxerr(c, ct, M,N);
    if(debug)
    {
        for (int j = 0; j < C.height(); j++) {
            for (int i = 0; i < C.width(); i++) {
                printf("%.1f  ",C(i,j));
            }
            printf("\n");
        }
    }
    

    unsigned long t0, t1;
    float totalTime = 0;
    for (int i = 0; i < repeat_count; i++)
    {
        t0 = get_cur_time();
        gemm.realize(C);
        t1 = get_cur_time();
        totalTime += ((float)(t1 - t0) / 1000.);
    }
    printf("[rep %d] autokernel | blas \t%.4f ms \t",repeat_count, totalTime / repeat_count);


    totalTime = 0;
    for (int i = 0; i < repeat_count; i++)
    {
        t0 = get_cur_time();
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1, a, K, b, N, 0, ct, N),
        t1 = get_cur_time();
        totalTime += ((float)(t1 - t0) / 1000.);
    }
    printf("%.4f ms\n",totalTime / repeat_count);
    return 0;
}
