#include "Halide.h"
#include "HalideBuffer.h"
using namespace Halide;
using Halide::Expr;
using Halide::Func;
using Halide::Generator;
using Halide::Var;
using Halide::BoundaryConditions::constant_exterior;

class halide_normalize:public Halide::Generator<halide_normalize>{
public:
    // args
    Input<Buffer<float>> input{"input", 4};
    Input<Buffer<float>> scale{"scale", 1};
    Output<Buffer<float>> output{"output", 4};

    void generate()
    {
	    /* THE ALGORITHM */
        const Expr channel_number = input.dim(2).extent();

        Var n("n"), c("c"), h("h"), w("w");
        RDom cn(0, channel_number);
        Func channel_reduce("channel_reduce");
        channel_reduce(w, h, n) += input(w, h, cn, n) * input(w, h, cn, n);
        channel_reduce(w, h, n) = 1.f / sqrt(channel_reduce(w, h, n));

        Func halide_normalize("halide_normalize");
        halide_normalize(w, h, c, n) = channel_reduce(w, h, n) * scale(c) * input(w, h, c, n);

        output(w, h, c, n) = halide_normalize(w, h, c, n);
    }

    void schedule()
    {
	    /* THE SCHEDULE */
        input.set_estimates({{0, 512}, {0, 512}, {0, 512}, {0, 512}});
        scale.set_estimates({{0, 512}});
        output.set_estimates({{0, 512}, {0, 512}, {0, 512}, {0, 512}});
    }
};

HALIDE_REGISTER_GENERATOR(halide_normalize, halide_normalize)
