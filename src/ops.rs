use nalgebra::{ArrayStorage, Const, Matrix};

use crate::{Differentiable, Node};

#[derive(Clone)]
pub struct Add<Lhs, Rhs>(pub Lhs, pub Rhs);

impl<'a, L, R, LNode: Differentiable<'a, T = L>, RNode: Differentiable<'a, T = R>>
    Differentiable<'a> for Add<LNode, RNode>
where
    L: std::ops::Add<R>,
{
    type Δ = Add<LNode::Δ, RNode::Δ>;
    type T = <L as std::ops::Add<R>>::Output;

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

#[derive(Clone)]
pub struct Mul<Lhs, Rhs>(pub Lhs, pub Rhs);

// h(x) = w * x
// y = f(h(x))
//
// dh/dw = w * x' + w' * x
//
// dy/dw = f'(h(x)) * x^T
// dy/dx = w^T * f'(h(x))

impl<'a, L, R, LNode: Differentiable<'a, T = L> + 'a, RNode: Differentiable<'a, T = R> + 'a>
    Differentiable<'a> for Mul<LNode, RNode>
where
    L: std::ops::Mul<R>,
{
    type Δ = Add<Mul<&'a LNode, RNode::Δ>, Mul<&'a RNode, LNode::Δ>>;
    type T = <L as std::ops::Mul<R>>::Output;

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

macro_rules! impl_f32_op {
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

impl_f32_op!(Add:add Mul:mul);
impl_debug!(*Mul, +Add);
