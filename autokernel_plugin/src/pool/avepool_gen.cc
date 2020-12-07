#include "Halide.h"
#include "HalideBuffer.h"
using namespace Halide;
using Halide::Expr;
using Halide::Func;
using Halide::Generator;
using Halide::Var;
using Halide::BoundaryConditions::constant_exterior;

class halide_avepool:public Halide::Generator<halide_avepool>{
public:
    // args
    Input<Buffer<float>> input{"input", 4};
    Input<int> stride{"stride"};
    Input<int> pad_width{"pad_width"};
    Input<int> pad_height{"height"};
    Input<int> kernel_w{"kernel_w"};
    Input<int> kernel_h{"kernel_h"};
    Output<Buffer<float>> output{"output", 4};

    void generate()
    {
        /* THE ALGORITHM */
        Var x("x"), y("y"), c("c"), n("n");

        constexpr float kMinValue = -3.4028235e38;
        Func input_bounded = constant_exterior(input, kMinValue,
                                               {{0, input.dim(0).extent()},
                                                {0, input.dim(1).extent()},
                                                {Expr(), Expr()},
                                                {Expr(), Expr()},
                                                });
        Func input_padded("input_padded");
        input_padded(x, y, c, n) = input_bounded(x - pad_width, y - pad_height, c, n);

        Func sum("sum");
        RDom filter_dom(0, kernel_w, 0, kernel_h);
        sum(x, y, c, n) += select(
                                stride == 1,
                                input_padded(x + filter_dom.x, y + filter_dom.y, c, n),
                                input_padded(x * stride + filter_dom.x, y * stride + filter_dom.y, c, n) );
        Expr in_x_origin = x * stride - pad_width;
        Expr x_start = max(0, -in_x_origin);
        Expr x_end = min(kernel_w, input.dim(0).extent() - in_x_origin);

        Expr in_y_origin = y * stride - pad_height;
        Expr y_start = max(0, -in_y_origin);
        Expr y_end = min(kernel_h, input.dim(1).extent() - in_y_origin);

        Expr filter_count = (x_end - x_start) * (y_end - y_start);

        output(x, y, c, n) = sum(x, y, c, n) / filter_count;
    }

    void schedule()
    {
        if(auto_schedule)
        {
            input.set_estimates({{0, 512}, {0, 512}, {0, 512}, {0, 1}});
            stride.set_estimate(1);
            pad_width.set_estimate(1);
            pad_height.set_estimate(1);
            kernel_w.set_estimate(1);
            kernel_h.set_estimate(1);
            output.set_estimates({{0, 512}, {0, 512}, {0, 512}, {0, 1}});
        }
    }
};

HALIDE_REGISTER_GENERATOR(halide_avepool, halide_avepool)
