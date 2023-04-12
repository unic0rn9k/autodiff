use crate::{mat::MatrixNode, prelude::*, value, Node};
use nalgebra::{allocator::Allocator, DefaultAllocator, Dyn};

#[derive(Clone)]
pub struct Add<Lhs, Rhs>(pub Lhs, pub Rhs);

impl<'a, L, R, LNode: Differentiable<'a, T = L>, RNode: Differentiable<'a, T = R>>
    Differentiable<'a> for Add<LNode, RNode>
where
    L: std::ops::Add<R>,
{
    type Δ<D> = Add<LNode::Δ<D>, RNode::Δ<D>>where Self: 'a;
    type T = <L as std::ops::Add<R>>::Output;
    type Unit = LNode::Unit;

    fn eval(&self) -> Self::T {
        self.0.eval() + self.1.eval()
    }

    fn derivative<'d, const LEN: usize, D: Clone>(
        &'a self,
        k: [&str; LEN],d:D
    ) -> [Self::Δ<D>; LEN] {
        self.0
            .derivative(k.clone(), d.clone())
            .zip(self.1.derivative(k, d))
            .map(|(dx, dy)| Add(dx, dy))
    }

    fn is_zero(&self) -> bool {
        self.0.is_zero() && self.1.is_zero()
    }
}

impl<'a, L, R, LNode: Differentiable<'a, T = L>, RNode: Differentiable<'a, T = R>>
    Differentiable<'a> for Sub<LNode, RNode>
where
    L: std::ops::Sub<R>,
{
    type Δ<D> = Sub<LNode::Δ<D>, RNode::Δ<D>> where Self: 'a;
    type T = <L as std::ops::Sub<R>>::Output;
    type Unit = LNode::Unit;

    fn eval(&self) -> Self::T {
        self.0.eval() - self.1.eval()
    }

    fn derivative<const LEN: usize, D: Clone>(&'a self, k: [&str; LEN], d:D) -> [Self::Δ<D>; LEN] {
        self.0
            .derivative(k.clone(), d.clone())
            .zip(self.1.derivative(k,d))
            .map(|(dx, dy)| Sub(dx, dy))
    }

    fn is_zero(&self) -> bool {
        false
    }
}

#[derive(Clone)]
pub struct Sub<Lhs, Rhs>(pub Lhs, pub Rhs);

#[derive(Clone)]
pub struct Mul<Lhs, Rhs>(pub Lhs, pub Rhs);

impl<'a, L, R, LNode: Differentiable<'a, T = L> + 'a, RNode: Differentiable<'a, T = R> + 'a>
    Differentiable<'a> for Mul<LNode, RNode>
where
    L: std::ops::Mul<R>,
{
    type Δ<D> = Add<LNode::Δ<Mul<D, Transpose<&'a RNode>>>, RNode::Δ<Mul<Transpose<&'a LNode>, D>>>;
    type T = <L as std::ops::Mul<R>>::Output;
    type Unit = LNode::Unit;

    fn eval(&self) -> Self::T {
        self.0.eval() * self.1.eval()
    }

    fn derivative<const LEN: usize, D: Clone>(&'a self, k: [&str; LEN], d:D) -> [Self::Δ<D>; LEN] {
        let Mul(x, y) = self;
        let dx = self
            .0
            .derivative(k.clone(), Mul(d.clone(), Transpose(y)));
        let dy = self.1.derivative(k, Mul(Transpose(x), d));
        dx.zip(dy).map(|(dx, dy)| Add(dx, dy))
    }

    fn is_zero(&self) -> bool {
        self.0.is_zero() || self.1.is_zero()
    }
}

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
    type Unit = N::Unit;

    fn eval(&self) -> Self::T {
        self.0.eval().transpose_()
    }

    fn derivative<'d, const LEN: usize, D: Clone>(
        &'a self,
        k: [&str; LEN],
        d:D
    ) -> [Self::Δ<D>; LEN] {
        todo!()
    }

    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}

#[derive(Clone, Debug)]
pub struct Softmax<N>(pub N);

impl<'a, T: value::Scalar, N: Differentiable<'a, T = MatrixNode<T>> + 'a> Differentiable<'a>
    for Softmax<N>
where
    DefaultAllocator: Allocator<T, Dyn, Dyn>,
{
    type Δ<D> = N::Δ<Mul<MatrixNode<T>, D>>;
    type T = MatrixNode<T>;
    type Unit = N::Unit;

    fn eval(&self) -> Self::T {
        if let Some(m) = &self.0.eval().0 {
            let mut tmp = nalgebra::Matrix::<
                T,
                Dyn,
                Dyn,
                <DefaultAllocator as Allocator<T, Dyn, Dyn>>::Buffer,
            >::repeat(m.nrows(), m.ncols(), T::from(Zero));
            for mut row in 0..m.nrows() {
                let sum = m.row(row).iter().map(|x| x.exp()).sum::<T>();
                tmp.row_mut(row).iter_mut().for_each(|x| *x = x.exp() / sum);
            }
            MatrixNode(Some(tmp))
        } else {
            MatrixNode(None)
        }
    }

    fn derivative<'d, const LEN: usize, D: Clone>(
        &'a self,
        k: [&str; LEN],
        d:D
    ) -> [Self::Δ<D>; LEN] {
        let delta = if let Some(m) = &self.0.eval().0 {
            let mut tmp = nalgebra::Matrix::<
                T,
                Dyn,
                Dyn,
                <DefaultAllocator as Allocator<T, Dyn, Dyn>>::Buffer,
            >::repeat(m.nrows(), m.ncols(), T::from(Zero));
            for mut row in 0..m.nrows() {
                let sum = m.row(row).iter().map(|x| x.exp()).sum::<T>();
                tmp.row_mut(row).iter_mut().for_each(|x| {
                    let sm = x.exp() / sum;
                    *x = Mul(sm * (T::from(One) - sm), d.clone())
            });
            }
            MatrixNode(Some(tmp))
        } else {
            MatrixNode(None)
        }
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
