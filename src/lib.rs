#![feature(test, array_zip)]

use std::{fmt::Debug, marker::PhantomData};
mod symbol;

mod scalar;
#[cfg(test)]
mod test;
pub use symbol::{symbol, AnonymousSymbol, Symbol};
pub mod ops;

#[derive(Clone)]
pub struct Node<'a, N: Differentiable<'a>>(N, PhantomData<&'a ()>);

impl<'a, N: Debug + Differentiable<'a>> Debug for Node<'a, N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("@").field(&self.0).finish()
    }
}

impl<'a, N: Differentiable<'a>> Differentiable<'a> for Node<'a, N> {
    type Δ = N::Δ;
    type T = N::T;

    fn eval(&self) -> Self::T {
        self.0.eval()
    }

    fn derivative<const LEN: usize>(&'a self, k: &[&str; LEN]) -> [Self::Δ; LEN] {
        self.0.derivative(k)
    }
}

impl<'a, N: Differentiable<'a>> Node<'a, N> {
    fn new(node: N) -> Self {
        Self(node, PhantomData)
    }
}

pub trait Differentiable<'a> {
    type Δ;
    type T;

    fn eval(&self) -> Self::T;
    fn derivative<const LEN: usize>(&'a self, k: &[&str; LEN]) -> [Self::Δ; LEN];
    //fn id(self: &Arc<Self>) -> NodeId {
    //    NodeId(Arc::as_ptr(self) as *const ())
    //}
    fn symbol(self, symbol: &'static str) -> Node<'a, Symbol<'a, Self>>
    where
        Self: Sized,
        Self: Differentiable<'a, T = f32>,
        Self::Δ: Differentiable<'a, T = f32>,
    {
        Node::new(Symbol::new(self, symbol))
    }
}

impl<'a, N: Differentiable<'a>> Differentiable<'a> for &N {
    type Δ = N::Δ;
    type T = N::T;

    fn eval(&self) -> Self::T {
        (*self).eval()
    }

    fn derivative<const LEN: usize>(&'a self, k: &[&str; LEN]) -> [Self::Δ; LEN] {
        (*self).derivative(k)
    }
}

impl<'a> Differentiable<'a> for f32 {
    type Δ = f32;
    type T = f32;

    fn eval(&self) -> Self::T {
        *self
    }

    fn derivative<const LEN: usize>(&'a self, _: &[&str; LEN]) -> [Self::Δ; LEN] {
        [0.; LEN]
    }
}

#[test]
fn basic() {
    let x = 2f32.symbol("x");
    let y = 3f32.symbol("y");
    let f = &x * &y + &x * &x;

    let [dx, dy] = f.derivative(&["x", "y"]);

    println!("f  = {f:?}");
    println!("dx = {dx:?}");
    println!("dy = {dy:?}");

    assert_eq!(f.eval(), 10.);
    assert_eq!(dx.eval(), 7.);
    assert_eq!(dy.eval(), 2.);

    //let x = symbol("x");
    //let f = x * 1.2f32;
    //assert_eq!(f.derivative(&["x"])[0], 1.2);
}
