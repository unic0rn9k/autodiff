use crate::value::Atom;
use crate::{mat::MatrixNode, value::NodeValue};
use nalgebra::{allocator::Allocator, DefaultAllocator, Dyn};

pub trait Exp {
    type Output;
    fn exp(self) -> Self::Output;
}

pub trait ElemMul<Rhs = Self> {
    type Output;
    fn elem_mul(self, rhs: Rhs) -> Self::Output;
}

macro_rules! impl_ops {
    ($($t: ident $(:$exp:ident)?)*) => {$(
        impl ElemMul for $t {
            type Output = $t;
            fn elem_mul(self, rhs: Self) -> Self::Output {
                self * rhs
            }
        }

        impl ElemMul<MatrixNode<$t>> for NodeValue<$t> {
            type Output = MatrixNode<$t>;
            fn elem_mul(self, mut rhs: MatrixNode<$t>) -> Self::Output {
                rhs.0
                    .iter_mut()
                    .for_each(|m| m.iter_mut().for_each(|n| *n *= self.0));
                rhs
            }
        }

        impl ElemMul<NodeValue<$t>> for MatrixNode<$t> {
            type Output = MatrixNode<$t>;
            fn elem_mul(self, rhs: NodeValue<$t>) -> Self::Output {
                rhs.elem_mul(self)
            }
        }

        impl ElemMul<MatrixNode<$t>> for $t {
            type Output = MatrixNode<$t>;
            fn elem_mul(self, mut rhs: MatrixNode<$t>) -> Self::Output {
                rhs.0
                    .iter_mut()
                    .for_each(|m| m.iter_mut().for_each(|n| *n *= self));
                rhs
            }
        }

        impl ElemMul<$t> for MatrixNode<$t> {
            type Output = MatrixNode<$t>;
            fn elem_mul(self, rhs: $t) -> Self::Output {
                rhs.elem_mul(self)
            }
        }

        impl ElemMul<MatrixNode<$t>> for Atom {
            type Output = MatrixNode<$t>;
            fn elem_mul(self, mut rhs: MatrixNode<$t>) -> Self::Output {
                rhs.0
                    .iter_mut()
                    .for_each(|m| m.iter_mut().for_each(|n| *n *= $t::from(self)));
                rhs
            }
        }

        impl ElemMul<Atom> for MatrixNode<$t> {
            type Output = MatrixNode<$t>;
            fn elem_mul(self, rhs: Atom) -> Self::Output {
                rhs.elem_mul(self)
            }
        }

        $(impl Exp for $t {
            type Output = $t;
            fn $exp(self) -> Self::Output {
                $t::exp(self)
            }
        })?
    )*};
}

impl_ops!(f32:exp f64:exp u8 i8 u16 i16 u32 i32 u64 i64 u128 i128);

impl<T: Exp<Output = T> + Copy> Exp for MatrixNode<T>
where
    DefaultAllocator: Allocator<T, Dyn, Dyn>,
{
    type Output = Self;

    fn exp(mut self) -> Self::Output {
        self.0
            .iter_mut()
            .for_each(|m| m.iter_mut().for_each(|n| *n = n.exp()));
        self
    }
}

impl<L, R> ElemMul<MatrixNode<R>> for MatrixNode<L>
where
    L: ElemMul<R, Output = L> + Copy,
    R: Copy,
    DefaultAllocator: Allocator<L, Dyn, Dyn>,
    DefaultAllocator: Allocator<R, Dyn, Dyn>,
{
    type Output = Self;

    fn elem_mul(mut self, rhs: MatrixNode<R>) -> Self::Output {
        self.0.iter_mut().zip(rhs.0.iter()).for_each(|(m, n)| {
            m.iter_mut()
                .zip(n.iter())
                .for_each(|(a, b)| *a = a.elem_mul(*b))
        });
        self
    }
}
