#include "Halide.h"
#include "HalideBuffer.h"

using namespace Halide;
using Halide::BoundaryConditions::constant_exterior;
using Halide::Expr;

Var x("x"), y("y"), c("c"), n("n");

//conv with bias
class ConvGenerator : public Generator<ConvGenerator> {
public:
    Input<Buffer<float>> input{"input", 4}; //[w,h,c,n]
    Input<Buffer<float>> weight{"weight", 4}; //[kw,kh,cin,cout]
    Input<Buffer<float>> bias{"bias", 1}; //[cout]
    Input<int> stride{"stride"}; 
    Input<int> pad{"pad"}; 
    Output<Buffer<float>> output{"output", 4};

    void generate() {
        RDom r(0, weight.dim(0).extent(), 0, weight.dim(1).extent(), 0, weight.dim(2).extent()); 
        output(x, y, c, n) = bias(c);

        Func inp_bounded =constant_exterior(input,  //source
                                            0,      //value
                        {{0, input.dim(0).extent()},	    //boundary-dim0 w
                         {0, input.dim(1).extent()},	    //boundary-dim1 h
                         {Expr(), Expr()},                  //boundary-dim2 c
                         {Expr(), Expr()}});                //boundary-dim3 n
        Func inp_padded("inp_padded");
        inp_padded(x, y, c, n) = inp_bounded(x - pad, y - pad, c, n);

        output(x, y, c, n) += weight(r[0], r[1], r[2], c) * 
                inp_padded(x * stride + r[0], y * stride + r[1],r[2],n);
    }
};

//matmul
class MatMulGenerator : public Generator<MatMulGenerator> {
public:
    Input<Buffer<float>> input{"input", 2};
    Input<Buffer<float>> weight{"weight", 2};
    Input<Buffer<float>> bias{"bias", 1};
    Output<Buffer<float>> output{"output", 2};

    void generate() {
        RDom k(0, input.dim(0).extent());
        output(x, y) = bias(x);
        output(x, y) += input(k, y) * weight(x, k);
    }
};

// maxpool
class MaxPoolGenerator : public Generator<MaxPoolGenerator> {
public:
    Input<Buffer<float>> input_a{"input_a", 4};
    Input<int> ksize{"ksize"};
    Input<int> stride{"stride"};
    Output<Buffer<float>> output{"output", 4};

    void generate() {
        RDom r(0, ksize, 0, ksize);
        int pad = 0;
        output(x, y, c, n) = maximum(input_a(stride*x+ r.x -pad , stride*y+r.y - pad, c, n));
    }
};

// relu
class ReluGenerator : public Generator<ReluGenerator> {
public:
    
    Input<Buffer<float>> input_a{"input_a", 4};
    Output<Buffer<float>> output{"output", 4};

    void generate() {
        output(x, y, c, n) = max(0.0f,input_a(x,y,c,n));
    }
};
HALIDE_REGISTER_GENERATOR(MatMulGenerator, halide_matmul)
HALIDE_REGISTER_GENERATOR(MaxPoolGenerator, halide_maxpool)
HALIDE_REGISTER_GENERATOR(ReluGenerator, halide_relu)
HALIDE_REGISTER_GENERATOR(ConvGenerator, halide_conv)
