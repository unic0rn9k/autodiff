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

    let [dx, dy] = f.derivative(["x", "y"], 1f32);

    assert_eq!(dy.eval().0, 0.);
    assert!((dx.eval().0 - 1.).abs() < 1e-6, "{} != 1", dx.eval().0);
}

// # Expected output
// sum = exp(x_0) + exp(x_1) + ... + exp(x_n)
// softmax(x_n) = exp(x_n) / sum
//
// softmax'(x_n) = softmax(x_n) * (1 - softmax(x_n))
//
// f(x)/g(x) -> (f'(x)g(x) - f(x)g'(x)) / g(x)^2
// exp(x) / sum -> (exp(x) * sum - exp(x) * exp(x)) / sum^2
//
// # Actual output
// ((One
// * ((1.0
// / Sum(Exp(@("x"= [4x1] ))))
// .* Exp(@("x"= [4x1] ))))
// - Sum((One
// * (((Exp(@("x"= [4x1] ))
// / (Sum(Exp(@("x"= [4x1] )))
// .* Sum(Exp(@("x"= [4x1] )))))
// .* 1.0)
// .* Exp(@("x"= [4x1] ))))))
//
// = 1/sum * exp(x) - exp(x) / sum^2
#[test]
fn softmax() {
    use crate::prelude::*;

    let x = mat(nalgebra::DMatrix::from_column_slice(2, 1, &[1f32, 2f32])).symbol("x");
    let xm = mat(nalgebra::DMatrix::from_column_slice(2, 1, &[1f32, 0f32])).symbol("x");
    let ym = mat(nalgebra::DMatrix::from_column_slice(2, 1, &[0f32, 1f32])).symbol("x");
    let mm = mat(nalgebra::DMatrix::from_column_slice(2, 1, &[1f32, 1f32])).symbol("x");

    let x1 = Node(Exp(1f32.symbol("x1")));
    let x2 = Node(Exp(2f32.symbol("x2")));

    assert_eq!(
        Exp(&x).eval().0.unwrap().as_slice(),
        &[x1.eval().0, x2.eval().0]
    );

    let sum = &x1 + &x2;
    let sm1 = &x1 / &sum;
    let sm2 = &x2 / &sum;
    let [dsm1] = sm1.derivative(["x1"], 1f32);
    let [dsm2] = sm2.derivative(["x2"], 1f32);
    let dsm = [dsm1.eval().0, dsm2.eval().0];

    let sum = Sum(Exp(&x));
    let softmax = Node(crate::ops::Div(Exp(&x), &sum));

    println!("{:?}", dsm1);

    //println!("{:?}", softmax.0 .0.eval() / softmax.0 .1.eval());

    let [dsoftmax] = softmax.derivative(["x"], mm);
    let [ds1] = softmax.derivative(["x"], xm);
    let [ds2] = softmax.derivative(["x"], ym);

    //println!("{:?}", sum.eval_().0);
    //println!("{:?}", softmax.eval_().0.unwrap());
    //println!("{:?}", dsoftmax.eval_().0.unwrap());
    //println!();
    //println!("a: {dsoftmax:?}");
    //println!();
    //println!("b: {dsm1:?}");
    //println!();
    //println!("[{:?}, {:?}]", ds1.eval_().0, ds2.eval_().0);

    dsoftmax.eval();

    // 0.03103084923581511
    // 0.0795501864530196
    // 0.1807693485836927
    // 0.22928868580089717

    //let x = mat(nalgebra::DMatrix::from_column_slice(
    //    4,
    //    1,
    //    &[1f32, 2., 3., 4.],
    //));

    //crate::primitive_ops::Exp::exp(x);
}

#[test]
fn exp() {
    use crate::prelude::*;

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

    println!("{dx:?}");
    println!("{dy:?}");
    println!("{:?}", dx.eval().0);
    println!("{:?}", dy.eval().0);
}
