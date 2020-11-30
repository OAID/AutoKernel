#include "Halide.h"
#include <vector>

using namespace Halide;
namespace {

// Generator class for BLAS gemm operations.
template<class T>
class GEMMGenerator : public Generator<GEMMGenerator<T>> {
public:
    typedef Generator<GEMMGenerator<T>> Base;
    using Base::get_target;
    using Base::natural_vector_size;
    using Base::target;
    template<typename T2>
    using Input = typename Base::template Input<T2>;
    template<typename T2>
    using Output = typename Base::template Output<T2>;

    Input<Buffer<T>> A_ = {"A_", 2};
    Input<Buffer<T>> B_ = {"B_", 2};
    Output<Buffer<T>> result_ = {"result", 2};

    void generate() {

        const Expr num_rows = A_.height(); //M  A(K,M)
        const Expr num_cols = B_.width(); //N  B(N,k)
        const Expr sum_size = A_.width(); //K

        const int vec = 8;
        const int s = vec * 2;

        Input<Buffer<T>> *A_in = &A_;
        Input<Buffer<T>> *B_in = &B_;


        Var i, j, ii, ji, jii, iii, io, jo, t;
        Var ti[3], tj[3];

        Func A("A"), B("B"), Btmp("Btmp"), As("As"), Atmp("Atmp"), Bs("Bs");
        Btmp(i, j) = BoundaryConditions::constant_exterior(*B_in, cast<T>(0))(i, j);

        Bs(i, j, io) = Btmp(io * s + i, j);
        B(i, j) = Bs(i % s, j, i / s);

        Atmp(i, j) = (*A_in)(i, j);
        A(i, j) = Atmp(i, j);

        Var k("k");
        Func prod;
        prod(k, j, i) = A(k, i) * B(j, k);

        Func AB("AB");
        RDom rv(0, sum_size);
        AB(i, j) += prod(rv, i, j);

   
        result_(i, j) =  AB(i, j);

        //schedule
        result_.tile(i, j, ti[1], tj[1], i, j, 2 * s, 2 * s, TailStrategy::GuardWithIf);
        result_
            .tile(i, j, ii, ji, s, 4)
            .tile(i, j, ti[0], tj[0], i, j, 1, s / 4);

        result_.specialize(num_rows >= 512 && num_cols >= 512)
            .fuse(tj[1], ti[1], t)
            .parallel(t);

        result_.specialize(num_rows >= 128 && num_cols >= 128)
            .tile(ti[1], tj[1], ti[2], tj[2], ti[1], tj[1], 2, 2)
            .fuse(tj[2], ti[2], t)
            .parallel(t);

        //long N
        result_.specialize(num_rows >= 64 && num_cols >= 8000)
            .parallel(ti[1],4);
        result_.specialize(num_rows >= 64 && num_cols >= 256)
            .tile(ti[1], tj[1], ti[2], tj[2], ti[1], tj[1], 4, 2)
            .parallel(ti[2]);
        result_.specialize(num_rows >= 64 && num_cols >= 128)
            .tile(ti[1], tj[1], ti[2], tj[2], ti[1], tj[1], 2, 2)
            .fuse(tj[2], ti[2], t)
            .parallel(t);
        //long M
       result_.specialize(num_rows >= 512 && num_cols >=32)
            .tile(ti[1], tj[1], ti[2], tj[2], ti[1], tj[1], 1, 4)
            .parallel(tj[2]);
        // long N
        result_.specialize(num_rows >= 32 && num_cols >= 8000)
            .parallel(ti[1],8);
        result_.specialize(num_rows >= 16 && num_cols >= 256)
            .tile(ti[1], tj[1], ti[2], tj[2], ti[1], tj[1], 4, 1)
            .parallel(ti[2]);
        result_.specialize(num_rows >= 16 && num_cols >= 128)
            .tile(ti[1], tj[1], ti[2], tj[2], ti[1], tj[1], 2, 1)
            .fuse(tj[2], ti[2], t)
            .parallel(t);
 
        //
        result_.rename(tj[0], t);
        result_.bound(i, 0, num_cols).bound(j, 0, num_rows);

        AB.compute_at(result_, i)
            .bound_extent(j, 4)
            .unroll(j)
            .bound_extent(i, s)
            .vectorize(i)
            .update()
            .reorder(i, j, rv)
            .unroll(j)
            .unroll(rv, 2)
            .vectorize(i);

        Bs.compute_root()
            .split(j, jo, ji, s)
            .reorder(i, ji, io, jo)
            .unroll(i)
            .vectorize(ji);
        Bs.specialize(B_.width() >= 256 && B_.height() >= 64)
            .parallel(jo, 4);

        Btmp.compute_at(Bs, io)
            .vectorize(i)
            .unroll(j);

        A_.dim(0).set_min(0).dim(1).set_min(0);
        B_.dim(0).set_min(0).dim(1).set_bounds(0, sum_size);
        result_.dim(0).set_bounds(0, num_cols).dim(1).set_bounds(0, num_rows);
    }
};

}  // namespace

HALIDE_REGISTER_GENERATOR(GEMMGenerator<float>, halide_im2col_conv)

