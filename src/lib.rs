//! # Automatic Differentiation
//! A crate for automatic differentiation in Rust, with performant CPU and GPU backends, and a simple math-envy API.
//!
//! # MVP
//! - [ ] impl matmul on gpu.
//! - [ ] optimizations on compiled graph for equivalent expressions.
//! - [X] optimize basic equivalent expressions.
//! - [ ] Better pattern for getting and setting data in graph.
//! - [ ] Miri CI
//!
//! # Operations
//! - Matmul
//! - Sub
//! - Div
//! - Transpose
//! - Exp
//! - Sum
//! - Append
//! - Slice
//! - Hard mask (could be implemented in an iterator like fashion, and thus would also cover Slice)

#![feature(return_position_impl_trait_in_trait)]

pub mod backend;
pub mod derivatives;
pub mod error;
pub mod expr;
pub mod graph;
pub mod ops;

#[cfg(test)]
fn matmul(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0; m * k];
    for i in 0..m {
        for j in 0..k {
            for l in 0..n {
                c[i * k + j] += a[i * n + l] * b[l * k + j];
            }
        }
    }
    c
}

#[test]
fn basic_scalar() {
    use backend::*;
    use graph::*;
    use ops::*;

    let mut graph: Graph<f32, CpuHeap> = graph::Graph::new(CpuHeap);

    let x = graph.scalar(1.0, "x").unwrap();
    let y = graph.scalar(2.0, "y").unwrap();

    let z = (x + y) * (y + x + y);

    let z = z.compile(&mut graph);
    let z = graph.eval(&z);
    assert_eq!(graph.get_scalar(&z).unwrap(), 15., "{graph:?}");
}

#[test]
fn basic_matrix() {
    use backend::*;
    use graph::*;
    use ops::*;
    use rand::random;

    let mut graph: Graph<f32, CpuHeap> = graph::Graph::new(CpuHeap);

    let mut mata: Vec<f32> = (0..12).map(|_| random()).collect();
    let mut matb: Vec<f32> = (0..20).map(|_| random()).collect();

    let ma = graph.matrix::<3, 4>(mata).unwrap();
    let mb = graph.matrix::<4, 5>(matb).unwrap();

    let mc = ma.matmul(mb);
    let mc = mc.compile(&mut graph);

    for _ in 0..500 {
        mata = (0..12).map(|_| random()).collect();
        matb = (0..20).map(|_| random()).collect();
        let matc = matmul(&mata, &matb, 3, 4, 5);

        graph.get_matrix_mut(&ma.0).copy_from_slice(&mata);
        graph.get_matrix_mut(&mb.0).copy_from_slice(&matb);

        let mc = graph.eval(&mc);
        assert!(
            matc.iter()
                .zip(graph.get_matrix(&mc).unwrap().iter())
                .map(|(a, b)| (a - b).abs())
                .sum::<f32>()
                < 1e-6
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda() -> error::Result<()> {
    let dev = cudarc::driver::CudaDevice::new(0)?;

    // allocate buffers
    let inp = dev.htod_copy(vec![1.0f32; 100])?;
    let mut out = dev.alloc_zeros::<f32>(100)?;
    Ok(())
}
