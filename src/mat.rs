use nalgebra::Const;
pub use nalgebra::{Matrix, Storage};

use crate::Differentiable;

struct Float<T>(T);

pub struct MatrixNode<T, const R: usize, const C: usize, S: Storage<T, Const<R>, Const<C>>>(
    Matrix<T, Const<R>, Const<C>, S>,
);

impl<'a, T: 'a, const R: usize, const C: usize, S: Storage<T, Const<R>, Const<C>> + 'a>
    Differentiable<'a> for MatrixNode<T, R, C, S>
{
    type Δ = Float<T>;
    type T = &'a MatrixNode<T, R, C, S>;

    fn eval(&'a self) -> Self::T {
        self
    }

    fn derivative<const LEN: usize>(&'a self, k: &[&str; LEN]) -> [Self::Δ; LEN] {
        todo!()
    }
}
