use nalgebra::{ArrayStorage, Const, Matrix};

use crate::{mat::MatrixNode, prelude::*, Node};

#[derive(Clone)]
pub struct Add<Lhs, Rhs>(pub Lhs, pub Rhs);

impl<'a, L, R, LNode: Differentiable<'a, T = L>, RNode: Differentiable<'a, T = R>>
    Differentiable<'a> for Add<LNode, RNode>
where
    L: std::ops::Add<R>,
{
    type Δ<D> = Add<LNode::Δ<D>, RNode::Δ<D>>where Self: 'a;
    type T = <L as std::ops::Add<R>>::Output;

    fn eval(&self) -> Self::T {
        self.0.eval() + self.1.eval()
    }

    fn derivative<'d, const LEN: usize, D: Clone>(
        &'a self,
        k: [(&str, D); LEN],
    ) -> [Self::Δ<D>; LEN] {
        self.0
            .derivative(k.clone())
            .zip(self.1.derivative(k))
            .map(|(dx, dy)| Add(dx, dy))
    }

    fn is_zero(&self) -> bool {
        self.0.is_zero() && self.1.is_zero()
    }
}

#[derive(Clone)]
pub struct Mul<Lhs, Rhs>(pub Lhs, pub Rhs);

// h(x) = w * x
// y = f(h(x))
//
// dh/dw = w * x' + w' * x
//
// dy/dx = w^T * f'(h(x))
// dy/dw = f'(h(x)) * x^T
//
// w2 * l1
// l' = x^T
// r' = 0
//
// l * l' + r' * r
//
// L^T * R' + L' * R^T
//
// w2 * (w1 * x)
//
// rhs  = (w1 * x)
// rhs' = x
//
// lhs  = w2
// lhs' = 0
//
// lhs_inner    + rhs * lhs'T
// (rhs'T + lhs') * lhs +
//
// dx = w1^T * w2
//
// rhs' = w1
// lhs = w2

// x = [i, m]
// y = [j, n]^T
//
// a(i) = i * j
// b(m) = m * n
//
// x * y = [a(i), b(m)]
//
// ChatGPT says:
// (dC/dx) = (dA/dx)B + A(dB/dx)

// w2 * (w1 * x)
//
// ∇w2     = (w1 * x)T
// ∇(w1*x) = w2T
// ∇w1     = ∇(w1*x) * XT
// ∇x      = w1T * ∇w2

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

    fn derivative<const LEN: usize, D: Clone>(&'a self, k: [(&str, D); LEN]) -> [Self::Δ<D>; LEN] {
        let Mul(x, y) = self;
        let dx = self
            .0
            .derivative(k.clone().map(|(k, d)| (k, Mul(d, Transpose(y)))));
        let dy = self.1.derivative(k.map(|(k, d)| (k, Mul(Transpose(x), d))));
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
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<T, nalgebra::Dyn, nalgebra::Dyn>,
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
        k: [(&str, D); LEN],
    ) -> [Self::Δ<D>; LEN] {
        self.0.derivative(k).map(Transpose)
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
