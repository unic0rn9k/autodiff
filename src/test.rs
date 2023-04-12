use cudarc::{
    cublas::{sys::cublasOperation_t::CUBLAS_OP_N, CudaBlas, Gemm, GemmConfig},
    driver::safe::CudaDevice,
};

use flamer::flame;
extern crate test;
use crate::Differentiable;
use test::{black_box, Bencher};

#[flame]
#[bench]
fn basic(b: &mut Bencher) {
    let x = black_box(2f32.symbol("x"));
    let y = black_box(3f32.symbol("y"));
    b.iter(|| {
        black_box(&x * &y + &x * &x).eval();
    });
}

#[bench]
fn hand_written(b: &mut Bencher) {
    b.iter(|| {
        let x = black_box(2f32);
        let y = black_box(3f32);
        black_box(x * x + x * y);
    });
}

// The transa and transb parameters are used to specify whether the matrices A and B should be transposed before the multiplication. This can be useful in certain cases to optimize performance or to match the storage format of the matrices.
// The m, n, and k parameters are used to specify the dimensions of the matrices involved in the multiplication. Specifically, m and n specify the dimensions of the resulting matrix C, while k specifies the number of columns in A (or rows in B). These parameters are necessary for the algorithm to correctly compute the matrix multiplication.
// The lda, ldb, and ldc parameters are used to specify the leading dimension of the matrices A, B, and C, respectively. The leading dimension is the number of elements between successive rows of the matrix, and it is used to account for padding that may be added to the matrix to achieve better memory alignment.

#[bench]
fn cuda_mm(bench: &mut Bencher) {
    let dev = CudaDevice::new(0).unwrap();

    let a = dev.alloc_zeros::<f32>(9).unwrap(); // 3x3
    let b = dev.alloc_zeros::<f32>(6).unwrap(); // 3x2
    let mut c = dev.alloc_zeros::<f32>(6).unwrap(); // 3x2

    let blas = CudaBlas::new(dev).unwrap();

    bench.iter(|| unsafe {
        blas.gemm(
            GemmConfig {
                transa: CUBLAS_OP_N,
                transb: CUBLAS_OP_N,
                m: 3,
                n: 2,
                k: 3,
                alpha: 1.,
                lda: 3,
                ldb: 3,
                beta: 0.,
                ldc: 3,
            },
            &a,
            &b,
            &mut c,
        )
        .unwrap();
    })
}
