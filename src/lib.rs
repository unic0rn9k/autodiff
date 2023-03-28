#![feature(test, array_zip)]

use std::fmt::Debug;
mod symbol;

mod mat;
pub mod prelude;
#[cfg(test)]
mod test;
mod value;
use ops::Transpose;
pub use symbol::{symbol, Symbol};
pub mod ops;

#[derive(Clone)]
pub struct Node<N>(pub N);

impl<N: Debug> Debug for Node<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("@").field(&self.0).finish()
    }
}

impl<'a, N: Differentiable<'a>> Differentiable<'a> for Node<N> {
    type Δ<D> = N::Δ<D> where Self: 'a;
    type T = N::T;

    fn eval(&self) -> Self::T {
        self.0.eval()
    }

    fn derivative<'d, const LEN: usize, D: Clone>(
        &'a self,
        k: [(&str, D); LEN],
    ) -> [Self::Δ<D>; LEN] {
        self.0.derivative(k)
    }
}

impl<'a, N: Differentiable<'a>> Node<N> {
    fn new(node: N) -> Self {
        Self(node)
    }
}

pub trait Differentiable<'a> {
    type Δ<T>
    where
        Self: 'a;
    type T;

    fn eval(&self) -> Self::T;
    fn derivative<const LEN: usize, D: Clone>(&'a self, k: [(&str, D); LEN]) -> [Self::Δ<D>; LEN];

    fn symbol(self, symbol: &'static str) -> Node<Symbol<Self>>
    where
        Self: Sized + 'a,
    {
        Node::new(Symbol::new(self, symbol))
    }

    fn transpose(self) -> Transpose<Self>
    where
        Self: Sized,
    {
        Transpose(self)
    }
}

impl<'a, N: Differentiable<'a>> Differentiable<'a> for &N {
    type Δ<D> = N::Δ<D> where Self: 'a;
    type T = N::T;

    fn eval(&self) -> Self::T {
        (*self).eval()
    }

    fn derivative<'d, const LEN: usize, D: Clone>(
        &'a self,
        k: [(&str, D); LEN],
    ) -> [Self::Δ<D>; LEN] {
        (*self).derivative(k)
    }
}

#[test]
fn basic() {
    use crate::prelude::*;
    let x = 2f32.symbol("x");
    let y = 3f32.symbol("y");
    let f = &x * &y + &x * &x;

    let [dx, dy] = f.derivative([("x", One), ("y", One)]);

    println!("f  = {f:?}");
    println!("dx = {dx:?}");
    println!("dy = {dy:?}");

    assert_eq!(*f.eval(), 10.);
    //assert_eq!(Node(dx).eval(), 7.);
    //assert_eq!(Node(dy).eval(), 2.);

    //let x = symbol("x");
    //let f = x * 1.2f32;
    //assert_eq!(f.derivative(&["x"])[0], 1.2);
}
