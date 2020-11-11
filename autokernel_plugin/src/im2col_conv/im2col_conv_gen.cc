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

    GeneratorParam<bool> transpose_A_ = {"transpose_A", false};
    GeneratorParam<bool> transpose_B_ = {"transpose_B", false};

    // Standard ordering of parameters in GEMM functions.
    Input<T> a_ = {"a_", 1};
    Input<Buffer<T>> A_ = {"A_", 2};
    Input<Buffer<T>> B_ = {"B_", 2};
    Input<T> b_ = {"b_", 1};
    Input<Buffer<T>> C_ = {"C_", 2};

    Output<Buffer<T>> result_ = {"result", 2};

    void generate() {
        // Matrices are interpreted as column-major by default. The
        // transpose GeneratorParams are used to handle cases where
        // one or both is actually row major.
        // const Expr num_rows = A_.width();
        // const Expr num_cols = B_.height();
        // const Expr sum_size = A_.height();
        const Expr num_rows = A_.height();
        const Expr num_cols = B_.width();
        const Expr sum_size = A_.width();

        const int vec = natural_vector_size(a_.type());
        const int s = vec * 2;

        Input<Buffer<T>> *A_in = &A_;
        Input<Buffer<T>> *B_in = &B_;


        Var i, j, ii, ji, jii, iii, io, jo, t;
        Var ti[3], tj[3];

        // // Swizzle A for better memory order in the inner loop.
        // Func A("A"), B("B"), Btmp("Btmp"), As("As"), Atmp("Atmp");
        // Atmp(i, j) = BoundaryConditions::constant_exterior(*A_in, cast<T>(0))(i, j);

        // As(i, j, io) = Atmp(io * s + i, j);

        // A(i, j) = As(i % s, j, i / s);

        // Btmp(i, j) = (*B_in)(i, j);
        
        // B(i, j) = Btmp(i, j);

        Func A("A"), B("B"), Btmp("Btmp"), As("As"), Atmp("Atmp"), Bs("Bs");
        Btmp(i, j) = BoundaryConditions::constant_exterior(*B_in, cast<T>(0))(i, j);

        Bs(i, j, io) = Btmp(io * s + i, j);

        B(i, j) = Bs(i % s, j, i / s);

        Atmp(i, j) = (*A_in)(i, j);
        
        A(i, j) = Atmp(i, j);

        Var k("k");
        Func prod;
        // Express all the products we need to do a matrix multiply as a 3D Func.
        // prod(k, i, j) = A(i, k) * B(k, j);
        prod(k, j, i) = A(k, i) * B(j, k);

        // Reduce the products along k.
        Func AB("AB");
        RDom rv(0, sum_size);
        AB(i, j) += prod(rv, i, j);

        Func ABt("ABt");
        
        ABt(i, j) = AB(i, j);
            

        // Do the part that makes it a 'general' matrix multiply.
        result_(i, j) = (a_ * ABt(i, j) + b_ * C_(i, j));

        result_.tile(i, j, ti[1], tj[1], i, j, 2 * s, 2 * s, TailStrategy::GuardWithIf);
        result_
            .tile(i, j, ii, ji, s, 4)
            .tile(i, j, ti[0], tj[0], i, j, 1, s / 4);

        // // If we have enough work per task, parallelize over these tiles.
        result_.specialize(num_rows >= 512 && num_cols >= 512)
            .fuse(tj[1], ti[1], t)
            .parallel(t);

        // Otherwise tile one more time before parallelizing, or don't
        // parallelize at all.
        result_.specialize(num_rows >= 128 && num_cols >= 128)
            .tile(ti[1], tj[1], ti[2], tj[2], ti[1], tj[1], 2, 2)
            .fuse(tj[2], ti[2], t)
            .parallel(t);

        result_.rename(tj[0], t);

        result_.bound(i, 0, num_cols).bound(j, 0, num_rows);

        Bs.compute_root()
            .split(j, jo, ji, s)
            .reorder(i, ji, io, jo)
            .unroll(i)
            .vectorize(ji)
            .specialize(B_.width() >= 256 && B_.height() >= 256)
            .parallel(jo, 4);

        Btmp.compute_at(Bs, io)
            .vectorize(i)
            .unroll(j);
/*
	AB.compute_at(result_, i)
	    .bound_extent(j, 4)
	    .unroll(j)
	    .bound_extent(i, s)
	    .vectorize(i)
	    .update()
	    .reorder(i, j, rv)
	    .unroll(j);
*/

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

        A_.dim(0).set_min(0).dim(1).set_min(0);
        B_.dim(0).set_min(0).dim(1).set_bounds(0, sum_size);
        C_.dim(1).set_bounds(0, num_rows);
        C_.dim(0).set_bounds(0, num_cols);
        result_.dim(0).set_bounds(0, num_cols).dim(1).set_bounds(0, num_rows);
    }
};

}  // namespace

HALIDE_REGISTER_GENERATOR(GEMMGenerator<float>, halide_im2col_conv)
//HALIDE_REGISTER_GENERATOR(GEMMGenerator<double>, dgemm)
