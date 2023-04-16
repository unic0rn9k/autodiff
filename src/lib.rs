//! A library for automatic differentiation. [![Rust](https://github.com/unic0rn9k/autodiff/actions/workflows/rust.yml/badge.svg)](https://github.com/unic0rn9k/autodiff/actions/workflows/rust.yml)
//!
//! # TODO
//! - [ ] separate `Mul` and `MatMul`
//! - [X] `.derivative()` should only take one gradient
//! - [ ] `exp` operator
//! - [ ] `sum` operator
//! - [ ] `div` operator
//!
//! ## softmax
//! sum = exp(x_0) + exp(x_1) + ... + exp(x_n)
//! softmax(x_n) = exp(x_n) / sum
//!
//! softmax'(x_n) = softmax(x_n) * (1 - softmax(x_n))
//!
//! f(x)/g(x) -> (f'(x)g(x) - f(x)g'(x)) / g(x)^2
//! exp(x) / sum -> (exp(x) * sum - exp(x) * exp(x)) / sum^2

#![feature(test, array_zip)]

use std::fmt::Debug;
mod symbol;

mod mat;
pub mod prelude;
pub mod primitive_ops;
#[cfg(test)]
mod test;
mod value;
use ops::Transpose;
pub use symbol::{symbol, Symbol};

use crate::ops::{ElemMul, Neg};
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
        k: [&str; LEN],
        d: D,
    ) -> [Self::Δ<D>; LEN] {
        self.0.derivative(k, d)
    }

    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}

impl<'a, N: Differentiable<'a>> Node<N> {
    fn new(node: N) -> Self {
        Self(node)
    }
}

pub trait Differentiable<'a> {
    type Δ<D>
    where
        Self: 'a;
    type T;

    fn eval(&self) -> Self::T;
    fn derivative<const LEN: usize, D: Clone>(&'a self, k: [&str; LEN], d: D)
        -> [Self::Δ<D>; LEN];

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

    fn is_zero(&self) -> bool;
}

impl<'a, N: Differentiable<'a>> Differentiable<'a> for &N {
    type Δ<D> = N::Δ<D> where Self: 'a;
    type T = N::T;

    fn eval(&self) -> Self::T {
        (*self).eval()
    }

    fn derivative<'d, const LEN: usize, D: Clone>(
        &'a self,
        k: [&str; LEN],
        d: D,
    ) -> [Self::Δ<D>; LEN] {
        (*self).derivative(k, d)
    }

    fn is_zero(&self) -> bool {
        (*self).is_zero()
    }
}

#[test]
fn basic() {
    use crate::prelude::*;
    let x = 2f32.symbol("x");
    let y = 3f32.symbol("y");
    let f = (&x * &y + &x * &x) / (&y + &x);
    //let f = &x / &y;

    let [dx, dy] = f.derivative(["x", "y"], 1f32);

    println!("f  = {f:?}");
    //println!("dx = {dx:?}");
    //println!("dy = {dy:?}");

    //assert_eq!(*f.eval(), 10.);
    //assert_eq!(Node(dx).eval(), 7.);
    //assert_eq!(Node(dy).eval(), 2.);

    //let x = symbol("x");
    //let f = x * 1.2f32;
    //assert_eq!(f.derivative(&["x"])[0], 1.2);

    assert_eq!(dy.eval(), 0.);
    assert!((dx.eval() - 1.).abs() < 1e-6, "{} != 1", dx.eval());

    //Neg(&y).eval();
}
