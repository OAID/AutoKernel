#include "Halide.h"
#include "HalideBuffer.h"
using namespace Halide;
using Halide::Expr;
using Halide::Func;
using Halide::Generator;
using Halide::Var;
using Halide::BoundaryConditions::constant_exterior;

class halide_softmax:public Halide::Generator<halide_softmax>{
public:
    // args
    Input<Buffer<float>> input{"input", 2};

    Output<Buffer<float>> output{"output", 2};

    void generate()
    {
	/* THE ALGORITHM */
    const Expr num_classes=input.width();
	Var in("in"), n("n");
	Func expInput;
    RDom r(0,num_classes);
	expInput(in, n) = exp(input(in, n));
    Expr globalSum=sum(expInput(r.x,n));


	output(in,n)=expInput(in,n)/globalSum;
    }

    void schedule()
    {
	/* THE SCHEDULE */
        input.set_estimates({{0, 512}, {0, 512}});
        output.set_estimates({{0, 512}, {0, 512}});
    }
};

HALIDE_REGISTER_GENERATOR(halide_softmax, halide_softmax)