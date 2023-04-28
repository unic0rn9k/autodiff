use crate::{mat::MatrixNode, prelude::*, Node};
/// This module contains the implementations for the various operators
use nalgebra::{allocator::Allocator, DefaultAllocator, Dyn};

fn zip_map<const LEN: usize, A, B, C>(a: [A; LEN], b: [B; LEN], f: impl Fn(A, B) -> C) -> [C; LEN] {
    match a
        .into_iter()
        .zip(b.into_iter())
        .map(|(a, b)| f(a, b))
        .collect::<Vec<_>>()
        .try_into()
    {
        Ok(v) => v,
        Err(_) => unreachable!(),
    }
}

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
        zip_map(
            self.0.derivative(k, d.clone()),
            self.1.derivative(k, d),
            Add,
        )
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
        zip_map(
            self.0.derivative(k, d.clone()),
            self.1.derivative(k, d),
            Sub,
        )
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
    // 1/y * dy/dx - x/y^2 * dy/dx
    type Δ<D> = Sub<LNode::Δ<Div<D, &'a RNode>>, RNode::Δ<ElemMul<Div<&'a LNode, ElemMul<&'a RNode, &'a RNode>>, D>>> where Self: 'a;

    type T = <L as std::ops::Div<R>>::Output;

    fn eval(&self) -> Self::T {
        self.0.eval() / self.1.eval()
    }

    fn derivative<const LEN: usize, D: Clone>(
        &'a self,
        k: [&str; LEN],
        d: D,
    ) -> [Self::Δ<D>; LEN] {
        let Div(x, y) = self;
        zip_map(
            self.0.derivative(k, Div(d.clone(), y)),
            self.1.derivative(k, ElemMul(Div(x, ElemMul(y, y)), d)),
            Sub,
        )
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
        let ElemMul(x, y) = self;
        zip_map(
            self.0.derivative(k, ElemMul(y, d.clone())),
            self.1.derivative(k, ElemMul(x, d)),
            Add,
        )
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
        zip_map(
            self.0.derivative(k, Mul(d.clone(), Transpose(y))),
            self.1.derivative(k, Mul(Transpose(x), d)),
            Add,
        )
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
        // Maybe transpose d here?
        self.0.derivative(k, d).map(Transpose)
    }

    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}

#[derive(Clone, Debug)]
pub struct Sum<N>(pub N);

impl<'a, T: Scalar + Copy + std::iter::Sum, N: Differentiable<'a, T = MatrixNode<T>>>
    Differentiable<'a> for Sum<N>
where
    DefaultAllocator: Allocator<T, Dyn, Dyn>,
{
    type Δ<D> = N::Δ<D> where Self: 'a;
    type T = NodeValue<T>;

    fn eval(&self) -> Self::T {
        NodeValue(
            self.0
                .eval()
                .0
                .map(|m| m.iter().copied().sum())
                .unwrap_or(T::from(Zero)),
        )
    }

    fn derivative<const LEN: usize, D: Clone>(
        &'a self,
        k: [&str; LEN],
        d: D,
    ) -> [Self::Δ<D>; LEN] {
        self.0.derivative(k, d)
    }

    fn is_zero(&self) -> bool {
        false
    }
}

#[derive(Clone, Debug)]
pub struct Exp<N>(pub N);

impl<'a, N: Differentiable<'a>> Differentiable<'a> for Exp<N>
where
    N::T: crate::primitive_ops::Exp,
{
    type Δ<D> = N::Δ<ElemMul<D,&'a Self>> where Self: 'a;

    type T = <N::T as crate::primitive_ops::Exp>::Output;

    fn eval(&self) -> Self::T {
        crate::primitive_ops::Exp::exp(self.0.eval())
    }

    fn derivative<const LEN: usize, D: Clone>(
        &'a self,
        k: [&str; LEN],
        d: D,
    ) -> [Self::Δ<D>; LEN] {
        self.0.derivative(k, ElemMul(d, self))
    }

    fn is_zero(&self) -> bool {
        false
    }
}

macro_rules! impl_debug {
    ($($op: literal $name: ident),*) => {
        $(impl<'a, Lhs: std::fmt::Debug, Rhs: std::fmt::Debug> std::fmt::Debug for $name<Lhs, Rhs> where Self: Differentiable<'a>{
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                if self.is_zero(){write!(f,"Zero")}else{
                    write!(f, "({:?} \n{} {:?})", self.0, $op, self.1)
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
impl_debug!("*" Mul, "+" Add, "-" Sub, "/" Div, ".*" ElemMul);
