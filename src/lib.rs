#![feature(test)]

use std::ops;
mod anon;

#[cfg(test)]
mod test;
pub use anon::{symbol, AnonymousSymbol};

use std::marker::PhantomData;

#[derive(Debug, Clone)]
pub struct DiffNode<'a, N: Differentiable<'a>> {
    //, const SYMBOL: &'static str> {
    //data: Option<N::T>,
    symbol: &'static str,
    node: N,
    marker: PhantomData<&'a ()>,
}

impl<'a, N: Differentiable<'a, T = f32>> Differentiable<'a> for DiffNode<'a, N>
where
    N::Δ: Differentiable<'a, T = f32>,
{
    type Δ = f32;
    type T = f32;

    fn eval(&self) -> Self::T {
        self.node.eval()
    }

    fn derivative(&'a self, k: &[&str]) -> Vec<Self::Δ> {
        k.iter()
            .map(|k| {
                if k == &self.symbol {
                    1.
                } else {
                    self.node.derivative(&[k])[0].eval()
                }
            })
            .collect()
    }
}

macro_rules! impl_self_and_ref {
    (impl<$($t:tt$(:$t2:path)?),*> $op:ty => for $self:ty => $($body:tt)*) => {
        impl<$($t$(:$t2)?),*> $op for $self $($body)*
        impl<$($t$(:$t2)?),*> $op for &'a $self $($body)*
    };
}

macro_rules! impl_f32_op {
    ($($Op:ident:$op: ident)*) => {
        $(impl_self_and_ref! {
            impl<'a, L: Differentiable<'a, T = f32>, R: Differentiable<'a, T = f32>> ops::$Op<R>
            => for DiffNode<'a, L> => where L:'a + Clone, R: 'a, L::Δ: Differentiable<'a, T=f32>,R::Δ: Differentiable<'a, T=f32>{
                type Output = DiffNode<'a, $Op<DiffNode<'a,L>, R>>;

                fn $op(self, rhs: R) -> Self::Output {
                    DiffNode{node:$Op(self.clone(), rhs),symbol:"",marker:PhantomData}
                }
            }
        })*
    };
}

impl_f32_op!(Add:add Mul:mul);

pub trait Differentiable<'a> {
    type Δ;
    type T;

    fn eval(&self) -> Self::T;
    fn derivative(&'a self, k: &[&str]) -> Vec<Self::Δ>;
    //fn id(self: &Arc<Self>) -> NodeId {
    //    NodeId(Arc::as_ptr(self) as *const ())
    //}
    fn symbol(self, symbol: &'static str) -> DiffNode<'a, Self>
    where
        Self: Sized,
    {
        DiffNode {
            symbol,
            node: self,
            marker: PhantomData,
        }
    }
}

impl<'a, N: Differentiable<'a>> Differentiable<'a> for &N {
    type Δ = N::Δ;
    type T = N::T;

    fn eval(&self) -> Self::T {
        (*self).eval()
    }

    fn derivative(&'a self, k: &[&str]) -> Vec<Self::Δ> {
        (*self).derivative(k)
    }
}

impl<'a> Differentiable<'a> for f32 {
    type Δ = f32;
    type T = f32;

    fn eval(&self) -> Self::T {
        *self
    }

    fn derivative(&'a self, k: &[&str]) -> Vec<Self::Δ> {
        k.iter().map(|_| 0.).collect()
    }
}

#[derive(Clone)]
pub struct Add<Lhs, Rhs>(Lhs, Rhs);

impl<'a, Lhs: Differentiable<'a, T = f32>, Rhs: Differentiable<'a, T = f32>> Differentiable<'a>
    for Add<Lhs, Rhs>
{
    type Δ = Add<Lhs::Δ, Rhs::Δ>;
    type T = f32;

    fn eval(&self) -> Self::T {
        self.0.eval() + self.1.eval()
    }

    fn derivative(&'a self, k: &[&str]) -> Vec<Self::Δ> {
        self.0
            .derivative(k)
            .drain(..)
            .zip(self.1.derivative(k).drain(..))
            .map(|(dx, dy)| Add(dx, dy))
            .collect()
    }
}

#[derive(Clone)]
pub struct Mul<Lhs, Rhs>(Lhs, Rhs);

impl<'a, Lhs: Differentiable<'a, T = f32> + 'a, Rhs: Differentiable<'a, T = f32> + 'a>
    Differentiable<'a> for Mul<Lhs, Rhs>
{
    type T = f32;
    type Δ = Add<Mul<&'a Lhs, Rhs::Δ>, Mul<&'a Rhs, Lhs::Δ>>;

    fn eval(&self) -> Self::T {
        self.0.eval() * self.1.eval()
    }

    fn derivative(&'a self, k: &[&str]) -> Vec<Self::Δ> {
        let mut dx = self.0.derivative(k);
        let mut dy = self.1.derivative(k);
        let Mul(x, y) = self;
        dx.drain(..)
            .zip(dy.drain(..))
            .map(|(dx, dy)| Add(Mul(x, dy), Mul(y, dx)))
            .collect()
    }
}

macro_rules! impl_debug {
    ($op: tt $name: ident) => {
        impl<Lhs: std::fmt::Debug, Rhs: std::fmt::Debug> std::fmt::Debug for $name<Lhs, Rhs> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "({:?} {} {:?})", self.0, stringify!($op), self.1)
            }
        }
    };
}

impl_debug!(*Mul);
impl_debug!(+Add);

#[test]
fn basic() {
    let x = 2f32.symbol("x");
    let y = 3f32.symbol("y");
    let f = &x * &y + &x * &x;

    let [dx, dy] = f.derivative(&["x", "y"])[..] else {unreachable!()};

    println!("dx = {dx:?}");
    println!("dy = {dy:?}");

    assert_eq!(f.eval(), 10.);
    assert_eq!(dx.eval(), 7.);
    assert_eq!(dy.eval(), 2.);

    //let x = symbol::<f32>("x");
    //let f = x * (1.2f32).symbol("");
    //f.derivate()
}
