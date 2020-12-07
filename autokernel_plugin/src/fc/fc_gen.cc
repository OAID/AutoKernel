#include "Halide.h"
#include "HalideBuffer.h"
using namespace Halide;
using Halide::Expr;
using Halide::Func;
using Halide::Generator;
using Halide::Var;
using Halide::BoundaryConditions::constant_exterior;

class halide_fc:public Halide::Generator<halide_fc>{
public:
    // args
    Input<Buffer<float>> input{"input", 2};
    Input<Buffer<float>> filter{"filter", 2};
    Input<Buffer<float>> bias{"bias", 1};
    Input<int> hidden{"hidden"};

    Output<Buffer<float>> output{"output", 2};

    void generate()
    {
	/* THE ALGORITHM */
	Var b("b"), co("co");
	Func halide_fc("halide_fc");
	RDom hi(0, hidden);
	halide_fc(co, b) = bias(co);
	halide_fc(co, b) += input(hi, b) * filter(hi, co);

	output(co, b) = halide_fc(co, b);
    }

    void schedule()
    {
	    /* THE SCHEDULE */
        input.set_estimates({{0, 512}, {0, 512}});
        kernel.set_estimates({{0, 512}, {0, 512}});
        bias.set_estimates({{0, 512}});

        hidden.set_estimate(1);
        output.set_estimates({{0, 512}, {0, 512}});
	
    }
};

HALIDE_REGISTER_GENERATOR(halide_fc, halide_fc)
