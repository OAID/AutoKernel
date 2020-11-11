#include "Halide.h"
#include "HalideBuffer.h"
using namespace Halide;
using Halide::Expr;
using Halide::Func;
using Halide::Generator;
using Halide::Var;
using Halide::BoundaryConditions::constant_exterior;

class halide_direct_conv:public Halide::Generator<halide_direct_conv>{
    public:
    Input<Buffer<float>> input{"input", 4};
    Input<Buffer<float>> kernel{"kernel", 4};
    Input<Buffer<float>> bias{"bias", 1};

    Input<int> input_c{"input_depth"};
    Input<int> stride{"stride"};
    Input<int> pad_width{"pad_width"};
    Input<int> pad_height{"pad_height"};
    Input<int> act{"act"};

    Output<Buffer<float>> relu{"relu", 4};

    void generate() {
        /* THE ALGORITHM */

        Var x("x"), y("y"), ci("ci"), n("n"), co("co");

	    Func inp_bounded =constant_exterior(input,	//source
			        0,			     	            //value
				{{0, input.dim(0).extent()},	    //boundary-dim0 w
	     		    	{0, input.dim(1).extent()},	    //boundary-dim1 h
            		    	{Expr(), Expr()},		    //boundary-dim2 c
	     		    	{Expr(), Expr()}});		    //boundary-dim3 n
        Func inp_padded("inp_padded");
        inp_padded(x, y, ci, n) = inp_bounded(x - pad_width, y - pad_height, ci, n);

        Func conv_nchw("conv_nchw");
        
        RDom r(0, kernel.dim(0).extent(), 0, kernel.dim(1).extent(), 0, input_c); 

        conv_nchw(x, y, co, n) = bias(co);
        conv_nchw(x, y, co, n) += kernel(r[0], r[1], r[2], co) * 
				inp_padded(x * stride + r[0], y * stride + r[1],r[2],n);

	relu(x, y, co, n) = select(act >= 0, max(act, conv_nchw(x, y, co, n)), conv_nchw(x, y, co, n));
	/*
	if(act == 0)
            relu(x, y, co, n) = max(act, conv_nchw(x, y, co, n));
	else
	    relu(x, y, co, n) = conv_nchw(x, y, co, n);
	*/
    }

    void schedule()
    {
	if(auto_schedule)
	{
	    input.set_estimates({{0, 512}, {0, 512}, {0, 512}, {0, 1}});
	    kernel.set_estimates({{0, 512}, {0, 512}, {0, 512}, {0, 512}});
	    bias.set_estimates({{0, 512}});
	    // input_c.set_estimate(64);
	    stride.set_estimate(1);
	    pad_width.set_estimate(1);
	    pad_height.set_estimate(1);
	    relu.set_estimates({{0, 512}, {0, 512}, {0, 512}, {0, 1}});
	}
    }


};
HALIDE_REGISTER_GENERATOR(halide_direct_conv, halide_direct_conv)
