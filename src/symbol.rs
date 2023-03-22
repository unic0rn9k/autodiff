use crate::{Differentiable, Node};
use std::{fmt::Debug, marker::PhantomData};

#[derive(Clone)]
pub struct Symbol<'a, N: Differentiable<'a>> {
    symbol: &'static str,
    node: N,
    marker: PhantomData<&'a ()>,
}

impl<'a, N: Differentiable<'a> + Debug> Debug for Symbol<'a, N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}={:?}", self.symbol, self.node)
    }
}

impl<'a, N: Differentiable<'a>> Symbol<'a, N> {
    pub fn new(node: N, symbol: &'static str) -> Self {
        Self {
            symbol,
            node,
            marker: PhantomData,
        }
    }
}

impl<'a, N: Differentiable<'a, T = f32>> Differentiable<'a> for Symbol<'a, N>
where
    N::Δ: Differentiable<'a, T = f32>,
{
    type Δ = f32;
    type T = f32;

    fn eval(&self) -> Self::T {
        self.node.eval()
    }

    fn derivative<const LEN: usize>(&'a self, k: &[&str; LEN]) -> [Self::Δ; LEN] {
        k.map(|k| {
            if k == self.symbol {
                1.
            } else {
                self.node.derivative(&[k])[0].eval()
            }
        })
    }
}

#[derive(Clone)]
pub struct AnonymousSymbol<T>(PhantomData<T>);

impl<'a> Differentiable<'a> for AnonymousSymbol<f32> {
    type Δ = f32;
    type T = f32;

    fn eval(&self) -> Self::T {
        0.
    }

    fn derivative<const LEN: usize>(&'a self, _: &[&str; LEN]) -> [Self::Δ; LEN] {
        [0.; LEN]
    }
}

pub fn symbol<'a>(s: &'static str) -> Node<'a, Symbol<'a, AnonymousSymbol<f32>>> {
    AnonymousSymbol(PhantomData).symbol(s)
}
