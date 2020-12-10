#include "Halide.h"
#include "HalideBuffer.h"
using namespace Halide;
using Halide::Expr;
using Halide::Func;
using Halide::Generator;
using Halide::Var;
using Halide::BoundaryConditions::constant_exterior;

class halide_depthwise:public Halide::Generator<halide_depthwise>{
public:
    Input<Buffer<float>> input{"input", 4};
    Input<Buffer<float>> kernel{"kernel", 4};
    Input<Buffer<float>> bias{"bias", 1};

    Input<int> stride{"stride"};
    Input<int> pad_width{"pad_width"};
    Input<int> pad_height{"pad_height"};
    Input<int> act{"act"};

    Output<Buffer<float>> output{"output", 4};

    void generate() {
        // The algorithm.
        Var x("x"), y("y"), depth("depth"), n("n");

        Func input_bounded =
            constant_exterior(input, 0,
                        {{0, input.dim(0).extent()},    //boundary-dim0 w
	     		    	{0, input.dim(1).extent()},	    //boundary-dim1 h
            		    {Expr(), Expr()},		        //boundary-dim2 c
	     		    	{Expr(), Expr()}});		        //boundary-dim3 n

        Func inp_padded("inp_padded");
        inp_padded(x, y, depth, n) = input_bounded(x - pad_width, y - pad_height, depth, n);

        Func conv_nchw("conv_nchw");        
        RDom filter_dom(0, kernel.dim(0).extent(), 0, kernel.dim(1).extent()); 

        conv_nchw(x, y, depth, n) = bias(depth);
        conv_nchw(x, y, depth, n) += kernel(filter_dom.x, filter_dom.y, 0, depth) *
             inp_padded(x * stride + filter_dom.x, y * stride + filter_dom.y, depth, n);		
       output(x, y, depth, n) = select(act >= 0, max(act, conv_nchw(x, y, depth, n)), conv_nchw(x, y, depth, n));
    }

    void schedule()
    {
        if(auto_schedule)
        {
            input.set_estimates({{0, 512}, {0, 512}, {0, 512}, {0, 1}});
            kernel.set_estimates({{0, 512}, {0, 512}, {0, 512}, {0, 1}});
            bias.set_estimates({{0, 512}});

            stride.set_estimate(1);
            pad_width.set_estimate(1);
            pad_height.set_estimate(1);
            output.set_estimates({{0, 512}, {0, 512}, {0, 512}, {0, 1}});
        }
    }
};

HALIDE_REGISTER_GENERATOR(halide_depthwise, halide_depthwise)
