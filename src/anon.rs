use std::marker::PhantomData;

use crate::{DiffNode, Differentiable};

#[derive(Clone)]
pub struct AnonymousSymbol<T>(PhantomData<T>);

impl<'a> Differentiable<'a> for AnonymousSymbol<f32> {
    type Δ = f32;
    type T = f32;

    fn eval(&self) -> Self::T {
        panic!("Anonymous symbol cannot be evaluated")
    }

    fn derivative(&'a self, k: &[&str]) -> Vec<Self::Δ> {
        vec![0.; k.len()]
    }
}

pub fn symbol<'a, T>(s: &'static str) -> DiffNode<'a, AnonymousSymbol<T>>
where
    AnonymousSymbol<T>: Differentiable<'a>,
{
    AnonymousSymbol(PhantomData).symbol(s)
}
