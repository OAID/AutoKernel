#include "Halide.h"
#include "HalideBuffer.h"
using namespace Halide;
using Halide::Expr;
using Halide::Func;
using Halide::Generator;
using Halide::Var;
using Halide::BoundaryConditions::constant_exterior;

class Halide_Func_Name:public Halide::Generator<Halide_Func_Name>{
public:
    // args
    Input<Buffer<float>> input{"input", 4};
    Input<int> param{"param"};

    Output<Buffer<float>> output{"output", 4};

    void generate()
    {
	/* THE ALGORITHM */
	Var x("x"), y("y"), c("c"), n("n");
	Func Halide_Func_Name("Halide_Func_Name");
	Halide_Func_Name(c, x, y, n) = input(c, x, y, n);

	output(c, x, y, n) = select(param >= 0, max(param, Halide_Func_Name(c, x, y, n)), Halide_Func_Name(c, x, y, n));
    }

    void schedule()
    {
	/* THE SCHEDULE */
    }
};

HALIDE_REGISTER_GENERATOR(Halide_Func_Name, Halide_Func_Name)
