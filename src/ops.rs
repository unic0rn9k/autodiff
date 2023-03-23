use nalgebra::{ArrayStorage, Const, Matrix};

use crate::{Differentiable, Node};
use std::marker::PhantomData;

/// When `TYPE='s'`, it is a scalar implementation. When it is 'm', it is a matrix implementation.
#[derive(Clone)]
pub struct Add<Lhs, Rhs, const TYPE: char = 's'>(pub Lhs, pub Rhs);

impl<'a, Lhs: Differentiable<'a, T = f32>, Rhs: Differentiable<'a, T = f32>> Differentiable<'a>
    for Add<Lhs, Rhs>
{
    type Δ = Add<Lhs::Δ, Rhs::Δ>;
    type T = f32;

    fn eval(&self) -> Self::T {
        self.0.eval() + self.1.eval()
    }

    fn derivative<const LEN: usize>(&'a self, k: &[&str; LEN]) -> [Self::Δ; LEN] {
        self.0
            .derivative(k)
            .zip(self.1.derivative(k))
            .map(|(dx, dy)| Add(dx, dy))
    }
}

impl<
        'a,
        S1: nalgebra::Storage<f32, nalgebra::Const<R>, nalgebra::Const<C>>,
        S2: nalgebra::Storage<f32, nalgebra::Const<R>, nalgebra::Const<C>>,
        const R: usize,
        const C: usize,
        Lhs: Differentiable<'a, T = Matrix<f32, Const<R>, Const<C>, S1>>,
        Rhs: Differentiable<'a, T = Matrix<f32, Const<R>, Const<C>, S2>>,
    > Differentiable<'a> for Add<Lhs, Rhs, 'm'>
{
    type Δ = Add<Lhs::Δ, Rhs::Δ>;
    type T = Matrix<f32, Const<R>, Const<C>, ArrayStorage<f32, R, C>>;

    fn eval(&self) -> Self::T {
        self.0.eval() + self.1.eval()
    }

    fn derivative<const LEN: usize>(&'a self, k: &[&str; LEN]) -> [Self::Δ; LEN] {
        self.0
            .derivative(k)
            .zip(self.1.derivative(k))
            .map(|(dx, dy)| Add(dx, dy))
    }
}

/// When `TYPE='s'`, it is a scalar implementation. When it is 'm', it is a matrix implementation.
#[derive(Clone)]
pub struct Mul<Lhs, Rhs, const TYPE: char = 's'>(pub Lhs, pub Rhs);

// h(x) = w * x
// y = f(h(x))
//
// dh/dw = w * x' + w' * x
//
// dy/dw = f'(h(x)) * x^T
// dy/dx = w^T * f'(h(x))
impl<
        'a,
        S1: nalgebra::Storage<f32, nalgebra::Const<K>, nalgebra::Const<L>>,
        S2: nalgebra::Storage<f32, nalgebra::Const<L>, nalgebra::Const<M>>,
        const K: usize,
        const L: usize,
        const M: usize,
        Lhs: Differentiable<'a, T = Matrix<f32, Const<K>, Const<L>, S1>>,
        Rhs: Differentiable<'a, T = Matrix<f32, Const<L>, Const<M>, S2>>,
    > Differentiable<'a> for Mul<Lhs, Rhs, 'm'>
{
    type Δ = Add<Lhs::Δ, Rhs::Δ>;
    type T = Matrix<f32, Const<K>, Const<M>, ArrayStorage<f32, K, M>>;

    fn eval(&self) -> Self::T {
        self.0.eval() * self.1.eval()
    }

    fn derivative<const LEN: usize>(&'a self, k: &[&str; LEN]) -> [Self::Δ; LEN] {
        todo!()
    }
}

impl<'a, Lhs: Differentiable<'a, T = f32> + 'a, Rhs: Differentiable<'a, T = f32> + 'a>
    Differentiable<'a> for Mul<Lhs, Rhs>
{
    type T = f32;
    type Δ = Add<Mul<&'a Lhs, Rhs::Δ>, Mul<&'a Rhs, Lhs::Δ>>;

    fn eval(&self) -> Self::T {
        self.0.eval() * self.1.eval()
    }

    fn derivative<const LEN: usize>(&'a self, k: &[&str; LEN]) -> [Self::Δ; LEN] {
        let dx = self.0.derivative(k);
        let dy = self.1.derivative(k);
        let Mul(x, y) = self;
        dx.zip(dy).map(|(dx, dy)| Add(Mul(x, dy), Mul(y, dx)))
    }
}

macro_rules! impl_debug {
    ($($op: tt $name: ident),*) => {
        $(impl<Lhs: std::fmt::Debug, Rhs: std::fmt::Debug> std::fmt::Debug for $name<Lhs, Rhs> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{:?} {} {:?}", self.0, stringify!($op), self.1)
            }
        })*
    };
}

macro_rules! impl_self_and_ref {
    (impl<'a, $($t:tt$(:$t2:path)?),*> $op:ty => for $self:ty => $($body:tt)*) => {
        impl<'a, $($t$(:$t2)?),*> $op for $self $($body)*
        impl<'a, $($t$(:$t2)?),*> $op for &'a $self $($body)*
    };
}

macro_rules! impl_f32_op {
    ($($Op:ident:$op: ident)*) => {
        $(impl_self_and_ref! {
            impl<'a, L: Differentiable<'a, T = f32>, R: Differentiable<'a, T = f32>> std::ops::$Op<R>
            => for Node<'a, L>
            => where L:'a + Clone, R: 'a, L::Δ: Differentiable<'a, T=f32>,R::Δ: Differentiable<'a, T=f32>{
                type Output = Node<'a, $Op<Node<'a, L>, R>>;

                fn $op(self, rhs: R) -> Self::Output {
                    Node($Op(self.clone(), rhs),PhantomData)
                }
            }
        })*
    };
}

impl_f32_op!(Add:add Mul:mul);
impl_debug!(*Mul, +Add);
