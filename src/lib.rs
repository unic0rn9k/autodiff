#![feature(test)]

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

    fn exp(&self) -> Node<crate::ops::Exp<&Self>> {
        Node(crate::ops::Exp(self))
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

    let [dx, dy] = f.derivative(["x", "y"], 1f32);

    assert_eq!(dy.eval().0, 0.);
    assert!((dx.eval().0 - 1.).abs() < 1e-6, "{} != 1", dx.eval().0);
}

#[test]
fn softmax() {
    use crate::prelude::*;

    fn delta_softmax(x: &mut [f32]) {
        let sum: f32 = x.iter().map(|x| x.exp().eval().0).sum();
        for x in x {
            let d = x.exp() / sum;
            *x = d * (1. - d)
        }
    }

    let mut x_src = [1f32, 2., 3., 4., 2.5, 3.8];

    let x = mat(nalgebra::DMatrix::from_column_slice(
        x_src.len(),
        1,
        &x_src.clone(),
    ))
    .symbol("x");

    let sum = Node(Sum(Exp(&x)));
    let softmax = crate::ops::Div(Exp(&x), &sum);
    let [dsoftmax] = softmax.derivative(["x"], 1f32);

    delta_softmax(&mut x_src);
    let dsoftmax_true = nalgebra::DMatrix::from_column_slice(x_src.len(), 1, &x_src);

    assert!((dsoftmax.eval().0.unwrap() - dsoftmax_true).abs().sum() < 1e-6);
}

#[test]
fn exp() {
    use crate::{ops::ElemMul, prelude::*};

    let x = mat(nalgebra::DMatrix::from_column_slice(
        4,
        1,
        &[1f32, 2., 3., 4.],
    ))
    .symbol("x");

    let exp = Node(Exp(&x * 4f32 + 1f32)) * 5f32;
    let dexp = ElemMul(20f32, Exp(&x * 4f32 + 1f32));

    let [dexp2] = exp.derivative(["x"], 1f32);
    assert_eq!(
        dexp.eval().0.unwrap().as_slice(),
        dexp2.eval().0.unwrap().as_slice()
    )
}

#[test]
fn div() {
    use crate::prelude::*;

    let x = 1f32.symbol("x");
    let y = 2f32.symbol("y");

    let f1 = &x / (&y + &x);
    let f2 = &y / (&y + &x);
    let [dx, _] = f1.derivative(["x", "y"], 1f32);
    let [_, dy] = f2.derivative(["x", "y"], 1f32);
    let dxv = dx.eval().0;
    let dxy = dy.eval().0;

    println!("{dx:?}");
    println!("{dy:?}");
    println!("{dxv:?}");
    println!("{dxy:?}");
    assert!(dxv - 0.2222222 < 1e-6);
    assert!(dxy - 0.1111111 < 1e-6);
}
