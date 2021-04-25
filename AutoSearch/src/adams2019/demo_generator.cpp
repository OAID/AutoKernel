#include "Halide.h"
#include "iostream"
//#include "utils.h"

namespace {

class BatchMatmul : public Halide::Generator<BatchMatmul> {
public:
    //std::vector<int> args = GetArgsFromEnv();
    int i = 0;
    //const int B = args[i++];
    //const int N = args[i++];
    //const int M = args[i++];
    //const int K = args[i++];
     const int B = 1;
     const int N = 1024;
     const int M = 1024;
     const int K = 1024;

    Input<Buffer<float>>    input_a{"input_a", 3};
    Input<Buffer<float>>    input_b{"input_b", 3};

    Output<Buffer<float>>   output{"output", 3};

    void generate() {
        Var x("x"), y("y"), b("b"),bo("bo"), xo("xo"), yo("yo"),xoa("xoa"),yoa("yoa"),yoai("yoai"),xi("xi");
        Var xii("xii");
        Var yi("yi");

        // Algorithm
        RDom k(0, K);


        Func func("func"), Bs("Bs");//,As("input_b_im#interleave");
        //As(x,y,xo,b) = input_b(xo*16+x,y,b);
        func(xi, y, b) = 0.0f;
        func(xi, y, b) += input_a(k, y, b) * input_b(xi,k,b);
        output(xi, y, b) = func(xi, y, b);
        //func.trace_stores();  

        output.bound(xi, 0, M)
              .bound(y, 0, N)
              .bound(b, 0, B);
        input_a.dim(0).set_bounds(0, K).set_stride(1)
               .dim(1).set_bounds(0, N).set_stride(K)
               .dim(2).set_bounds(0, B).set_stride(K * N);

        input_b.dim(0).set_bounds(0, M).set_stride(1)
               .dim(1).set_bounds(0, K).set_stride(M)
               .dim(2).set_bounds(0, B).set_stride(M * K);

        output.dim(0).set_bounds(0, M).set_stride(1)
              .dim(1).set_bounds(0, N).set_stride(M)
              .dim(2).set_bounds(0, B).set_stride(M * N);
        //Br.print_loop_nest();
        //func.print_loop_nest();
    }
};

}  // namespace

HALIDE_REGISTER_GENERATOR(BatchMatmul, demo)
