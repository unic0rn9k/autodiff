use crate::{mat::MatrixNode, prelude::*, Node};
/// This module contains the implementations for the various operators
use nalgebra::{allocator::Allocator, DefaultAllocator, Dyn};

#[derive(Clone)]
pub struct Add<Lhs, Rhs>(pub Lhs, pub Rhs);

impl<'a, L, R, LNode: Differentiable<'a, T = L>, RNode: Differentiable<'a, T = R>>
    Differentiable<'a> for Add<LNode, RNode>
where
    L: std::ops::Add<R>,
{
    type Δ<D> = Add<LNode::Δ<D>, RNode::Δ<D>> where Self: 'a;
    type T = <L as std::ops::Add<R>>::Output;

    fn eval(&self) -> Self::T {
        self.0.eval() + self.1.eval()
    }

    fn derivative<'d, const LEN: usize, D: Clone>(
        &'a self,
        k: [&str; LEN],
        d: D,
    ) -> [Self::Δ<D>; LEN] {
        self.0
            .derivative(k, d.clone())
            .zip(self.1.derivative(k, d))
            .map(|(dx, dy)| Add(dx, dy))
    }

    fn is_zero(&self) -> bool {
        self.0.is_zero() && self.1.is_zero()
    }
}

#[derive(Clone)]
pub struct Sub<Lhs, Rhs>(pub Lhs, pub Rhs);

impl<'a, L, R, LNode: Differentiable<'a, T = L>, RNode: Differentiable<'a, T = R>>
    Differentiable<'a> for Sub<LNode, RNode>
where
    L: std::ops::Sub<R>,
{
    type Δ<T> = Sub<LNode::Δ<T>, RNode::Δ<T>> where Self: 'a;

    type T = <L as std::ops::Sub<R>>::Output;

    fn eval(&self) -> Self::T {
        self.0.eval() - self.1.eval()
    }

    fn derivative<const LEN: usize, D: Clone>(
        &'a self,
        k: [&str; LEN],
        d: D,
    ) -> [Self::Δ<D>; LEN] {
        self.0
            .derivative(k, d.clone())
            .zip(self.1.derivative(k, d))
            .map(|(dx, dy)| Sub(dx, dy))
    }

    fn is_zero(&self) -> bool {
        false
    }
}

#[derive(Clone, Debug)]
pub struct Neg<N>(pub N);

impl<'a, N: Differentiable<'a>> Differentiable<'a> for Neg<N>
where
    N::T: std::ops::Neg,
{
    type Δ<D> = N::Δ<Neg<D>> where Self: 'a;
    type T = <N::T as std::ops::Neg>::Output;

    fn eval(&self) -> Self::T {
        -self.0.eval()
    }

    fn derivative<const LEN: usize, D: Clone>(
        &'a self,
        k: [&str; LEN],
        d: D,
    ) -> [Self::Δ<D>; LEN] {
        self.0.derivative(k, Neg(d))
    }

    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}

#[derive(Clone)]
pub struct Div<Lhs, Rhs>(pub Lhs, pub Rhs);

impl<'a, L, R, LNode: Differentiable<'a, T = L>, RNode: Differentiable<'a, T = R>>
    Differentiable<'a> for Div<LNode, RNode>
where
    L: std::ops::Div<R>,
{
    type Δ<D> = Add<LNode::Δ<Div<D, &'a RNode>>, RNode::Δ<ElemMul<Div<Neg<&'a LNode>, ElemMul<&'a RNode, &'a RNode>>, D>>> where Self: 'a;

    type T = <L as std::ops::Div<R>>::Output;

    fn eval(&self) -> Self::T {
        self.0.eval() / self.1.eval()
    }

    fn derivative<const LEN: usize, D: Clone>(
        &'a self,
        k: [&str; LEN],
        d: D,
    ) -> [Self::Δ<D>; LEN] {
        self.0
            .derivative(k, Div(d.clone(), &self.1))
            .zip(
                self.1
                    .derivative(k, ElemMul(Div(Neg(&self.0), ElemMul(&self.1, &self.1)), d)),
            )
            .map(|(dx, dy)| Add(dx, dy))
    }

    fn is_zero(&self) -> bool {
        false
    }
}

#[derive(Clone)]
pub struct ElemMul<Lhs, Rhs>(pub Lhs, pub Rhs);

impl<'a, L, R, LNode: Differentiable<'a, T = L> + 'a, RNode: Differentiable<'a, T = R> + 'a>
    Differentiable<'a> for ElemMul<LNode, RNode>
where
    L: crate::primitive_ops::ElemMul<R>,
{
    type Δ<D> = Add<LNode::Δ<ElemMul<&'a RNode, D>>, RNode::Δ<ElemMul<&'a LNode, D>>>
    where
        Self: 'a;

    type T = <L as crate::primitive_ops::ElemMul<R>>::Output;

    fn eval(&self) -> Self::T {
        self.0.eval().elem_mul(self.1.eval())
    }

    fn derivative<const LEN: usize, D: Clone>(
        &'a self,
        k: [&str; LEN],
        d: D,
    ) -> [Self::Δ<D>; LEN] {
        self.0
            .derivative(k, ElemMul(&self.1, d.clone()))
            .zip(self.1.derivative(k, ElemMul(&self.0, d)))
            .map(|(dx, dy)| Add(dx, dy))
    }

    fn is_zero(&self) -> bool {
        false
    }
}

#[derive(Clone)]
pub struct Mul<Lhs, Rhs>(pub Lhs, pub Rhs);

impl<'a, L, R, LNode: Differentiable<'a, T = L> + 'a, RNode: Differentiable<'a, T = R> + 'a>
    Differentiable<'a> for Mul<LNode, RNode>
where
    L: std::ops::Mul<R>,
{
    type Δ<D> = Add<LNode::Δ<Mul<D, Transpose<&'a RNode>>>, RNode::Δ<Mul<Transpose<&'a LNode>, D>>>;
    type T = <L as std::ops::Mul<R>>::Output;

    fn eval(&self) -> Self::T {
        self.0.eval() * self.1.eval()
    }

    fn derivative<const LEN: usize, D: Clone>(
        &'a self,
        k: [&str; LEN],
        d: D,
    ) -> [Self::Δ<D>; LEN] {
        let Mul(x, y) = self;
        let dx = self.0.derivative(k, Mul(d.clone(), Transpose(y)));
        let dy = self.1.derivative(k, Mul(Transpose(x), d));
        dx.zip(dy).map(|(dx, dy)| Add(dx, dy))
    }

    fn is_zero(&self) -> bool {
        self.0.is_zero() || self.1.is_zero()
    }
}

// w2 <- x*1
// w2 -> x
// + l1

trait TransposeAble {
    fn transpose_(self) -> Self;
}

impl<T: Scalar> TransposeAble for NodeValue<T> {
    fn transpose_(self) -> Self {
        self
    }
}

impl TransposeAble for Atom {
    fn transpose_(self) -> Self {
        self
    }
}

impl<T: Clone + PartialEq + std::fmt::Debug + 'static> TransposeAble for MatrixNode<T>
where
    DefaultAllocator: Allocator<T, Dyn, Dyn>,
{
    fn transpose_(self) -> Self {
        Self(self.0.map(|m| m.transpose()))
    }
}

#[derive(Clone, Debug)]
pub struct Transpose<N>(pub N);

impl<'a, T: TransposeAble, N: Differentiable<'a, T = T>> Differentiable<'a> for Transpose<N> {
    type Δ<D> = Transpose<N::Δ<D>>where Self: 'a;
    type T = T;

    fn eval(&self) -> Self::T {
        self.0.eval().transpose_()
    }

    fn derivative<'d, const LEN: usize, D: Clone>(
        &'a self,
        k: [&str; LEN],
        d: D,
    ) -> [Self::Δ<D>; LEN] {
        self.0.derivative(k, d).map(Transpose)
    }

    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}

macro_rules! impl_debug {
    ($($op: tt $name: ident),*) => {
        $(impl<'a, Lhs: std::fmt::Debug, Rhs: std::fmt::Debug> std::fmt::Debug for $name<Lhs, Rhs> where Self: Differentiable<'a>{
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                if self.is_zero(){write!(f,"Zero")}else{
                    write!(f, "({:?} {} {:?})", self.0, stringify!($op), self.1)
                }
            }
        })*
    };
}

macro_rules! impl_op {
    ($($Op:ident:$op: ident)*) => {
        $(
            impl<L, R> std::ops::$Op<R> for Node<L>
            {
                type Output = Node<$Op<Node<L>, R>>;

                fn $op(self, rhs: R) -> Self::Output {
                    Node($Op(self, rhs))
                }
            }

            impl<'a, L, R> std::ops::$Op<R> for &'a Node<L>
            {
                type Output = Node<$Op<&'a Node<L>, R>>;

                fn $op(self, rhs: R) -> Self::Output {
                    Node($Op(self, rhs))
                }
            }
        )*
    };
}

impl_op!(Add:add Mul:mul Sub:sub Div:div);
impl_debug!(*Mul, +Add, -Sub, /Div);
