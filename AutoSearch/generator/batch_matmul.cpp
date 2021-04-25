#include "Halide.h"
#include "iostream"
#include "utils.h"

/*
    C(x,y)+=A(k,y)*B(x,k)

                ____N____
             K  |   B   |
                |_______|

      __K___     ___N___
     |     |    |       |
    M|  A  |   M|   C   |
     |_____|    |_______|

*/
namespace {

class BatchMatmul : public Halide::Generator<BatchMatmul> {
public:
    std::vector<int> args = GetArgsFromEnv();
    int i = 0;
    const int B = args[i++];
    const int M = args[i++];
    const int N = args[i++];
    const int K = args[i++];

    // const int B = 1;
    // const int N = 1024;
    // const int N = 1024;
    // const int K = 1024;

    Input<Buffer<float>>    input_a{"input_a", 3}; //(dim0,dim1,dim2)=(width,heiht,batch)=(K,M,B)
    Input<Buffer<float>>    input_b{"input_b", 3}; //(dim0,dim1,dim2)=(width,heiht,batch)=(N,K,B)
    Output<Buffer<float>>   output{"output", 3};  //(dim0,dim1,dim2)=(width,heiht,batch)=(N,M,B)

    void generate() {
        Var x("x"), y("y"),b("b");
        RDom k(0, K);
        Func prod("prod"),Br("Br");
 
        // Algorithm
        prod(x, y, b) = 0.0f;
        prod(x, y, b) += input_a(k, y, b) * input_b(x, k, b);
        output(x, y, b) = prod(x, y, b);

        if (!auto_schedule) {
            Var xi("xi"), yi("yi"), xii("xii"), yii("yii"), xt("xt"), yt("yt"), xy("xy");

            if(get_target().has_gpu_feature())
            {
                // manuel gpu schedule
                output.tile(x,y,xi,yi,8,8)
                    .unroll(xi)
                    .unroll(yi)
                    .gpu_tile(x, y, xt, yt, 2, 2);

                prod.compute_at(output,x)
                    .gpu_threads(x,y)
                    .update()
                    .gpu_threads(x,y);
            }
            else 
            {
                //manuel cpu schedul
                output.tile(x, y, xi, yi, 16, 32)
                            .fuse(x, y, xy).parallel(xy)
                            .split(yi, yi, yii, 4)
                            .vectorize(xi, 8)
                            .unroll(xi)
                            .unroll(yii);

                prod.compute_at(output, yi)
                    .vectorize(x, 8).unroll(y);

                prod.update()
                    .reorder(x, y, k)
                    .vectorize(x, 8)
                    .unroll(x)
                    .unroll(y)
                    .unroll(k, 2);
            }
        }

        output.bound(x, 0, N)
              .bound(y, 0, M)
              .bound(b, 0, B);

        input_a.dim(0).set_bounds(0, K).set_stride(1)
               .dim(1).set_bounds(0, M).set_stride(K)
               .dim(2).set_bounds(0, B).set_stride(K * M);

        input_b.dim(0).set_bounds(0, N).set_stride(1)
               .dim(1).set_bounds(0, K).set_stride(N)
               .dim(2).set_bounds(0, B).set_stride(N * K);

        output.dim(0).set_bounds(0, N).set_stride(1)
              .dim(1).set_bounds(0, M).set_stride(N)
              .dim(2).set_bounds(0, B).set_stride(M * N);

    }
};

}  // namespace

HALIDE_REGISTER_GENERATOR(BatchMatmul, matmul)