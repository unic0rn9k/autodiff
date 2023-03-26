use crate::prelude::*;
use nalgebra::SMatrix;
use std::ops::*;

#[derive(Clone, Copy)]
pub struct MatrixNode<T, const R: usize, const C: usize>(pub Option<SMatrix<T, R, C>>);

impl<'a, T: Copy, const R: usize, const C: usize> Differentiable<'a> for SMatrix<T, R, C> {
    type Δ = Atom;
    type T = MatrixNode<T, R, C>;

    fn eval(&self) -> Self::T {
        MatrixNode(Some(*self))
    }

    fn derivative<const LEN: usize>(&'a self, _: &[&str; LEN]) -> [Self::Δ; LEN] {
        [Zero; LEN]
    }
}

//impl<'a, T: Copy, const R: usize, const C: usize> Differentiable<'a> for MatrixNode<T, R, C> {
//    type Δ = Atom;
//    type T = MatrixNode<T, R, C>;
//
//    fn eval(&self) -> Self::T {
//        *self
//    }
//
//    fn derivative<const LEN: usize>(&'a self, _: &[&str; LEN]) -> [Self::Δ; LEN] {
//        [Zero; LEN]
//    }
//}

impl<T, const R: usize, const C: usize> Add<MatrixNode<T, R, C>> for MatrixNode<T, R, C>
where
    SMatrix<T, R, C>: Add<SMatrix<T, R, C>, Output = SMatrix<T, R, C>>,
{
    type Output = MatrixNode<T, R, C>;

    fn add(self, rhs: MatrixNode<T, R, C>) -> Self::Output {
        MatrixNode(match (self.0, rhs.0) {
            (None, None) => None,
            (None, Some(r)) => Some(r),
            (Some(l), None) => Some(l),
            (Some(l), Some(r)) => Some(l + r),
        })
    }
}

impl<T, const K: usize, const L: usize, const M: usize> Mul<MatrixNode<T, M, L>>
    for MatrixNode<T, K, M>
where
    SMatrix<T, K, M>: Mul<SMatrix<T, M, L>, Output = SMatrix<T, K, L>>,
{
    type Output = MatrixNode<T, K, L>;

    fn mul(self, rhs: MatrixNode<T, M, L>) -> Self::Output {
        MatrixNode(match (self.0, rhs.0) {
            (Some(l), Some(r)) => Some(l * r),
            _ => None,
        })
    }
}

pub fn mat<T, const R: usize, const C: usize>(m: SMatrix<T, R, C>) -> MatrixNode<T, R, C> {
    MatrixNode(Some(m))
}

#[test]
fn gradient_decent() {
    let w = SMatrix::<f32, 2, 3>::new_random().symbol("weights");
    let b = SMatrix::<f32, 2, 1>::new_random().symbol("bias");

    let x = SMatrix::<f32, 1, 3>::new_random().symbol("input");

    let out = w * x; // + b;
    let [dw] = out.derivative(&["weights"]);

    assert_eq!(dw, x);
}
