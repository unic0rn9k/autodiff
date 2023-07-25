use crate::{backend::Backend, graph::Graph};

pub trait Differentiable {
    type T;

    fn derivative<'a, const LEN: usize, B: Backend<Self::T>>(
        &'a self,
        k: [&str; LEN],
        d: &'a impl Differentiable,
        b: &'a Graph<Self::T, B>,
    ) -> [impl Differentiable + 'a; LEN];
    //where
    //    Self: Op<Self::T, B>;
}

impl<T: Differentiable> Differentiable for &T {
    type T = T::T;

    fn derivative<'a, const LEN: usize, B: Backend<Self::T>>(
        &'a self,
        k: [&str; LEN],
        d: &'a impl Differentiable,
        b: &'a Graph<Self::T, B>,
    ) -> [impl Differentiable + 'a; LEN] {
        (**self).derivative(k, d, b)
    }
}
