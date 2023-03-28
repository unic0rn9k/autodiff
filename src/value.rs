use crate::Differentiable;
use nalgebra::{ArrayStorage, Const, Dim, SMatrix};
pub use nalgebra::{Matrix, Storage};
use std::ops::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Atom {
    Zero,
    One,
}
pub use Atom::*;

#[test]
fn ord() {
    assert!(One > Zero)
}

impl<'a> Differentiable<'a> for Atom {
    type Δ<T> = Atom;
    type T = Atom;

    fn eval(&self) -> Self::T {
        *self
    }

    fn derivative<const LEN: usize, D>(&'a self, _: [(&str, D); LEN]) -> [Self::Δ<D>; LEN] {
        [Zero; LEN]
    }
}

pub trait Scalar: From<Atom> {}

macro_rules! impl_scalar {
    ($($t: ty)*) => {$(
        impl From<Atom> for $t {
            fn from(n: Atom) -> Self {
                match n{
                    Zero => 0 as $t,
                    One  => 1 as $t
                }
            }
        }
        impl Scalar for $t{}

        impl<'a> Differentiable<'a> for $t{
                type Δ<T> = Atom;
                type T = NodeValue<$t>;

                fn eval(&self) -> Self::T {
                    NodeValue(*self)
                }

                fn derivative<const LEN: usize, D>(&'a self, _: [(&str, D); LEN]) -> [Self::Δ< D>; LEN] {
                    [Zero; LEN]
                }
        }
    )*};
}

impl_scalar!(u8 i8 u16 i16 u32 i32 u64 i64 u128 i128 f32 f64);

pub struct NodeValue<T>(pub T);

impl<T> Deref for NodeValue<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

macro_rules! impl_node_val_ops {
    ($($op:ident:$Op:ident $o:tt),*) => {$(
        impl<L: $Op<R>, R> $Op<NodeValue<R>> for NodeValue<L> {
            type Output = NodeValue<<L as $Op<R>>::Output>;

            fn $op(self, rhs: NodeValue<R>) -> Self::Output {
                NodeValue(self.0 $o rhs.0)
            }
        }

        impl<T: Scalar + $Op<Output=T>> $Op<Atom> for NodeValue<T>{
            type Output = T;

            fn $op(self, rhs: Atom) -> Self::Output {
                self.0 $o T::from(rhs)
            }
        }

        impl<T: Scalar + $Op<Output=T>> $Op<NodeValue<T>> for Atom{
            type Output = T;

            fn $op(self, rhs: NodeValue<T>) -> Self::Output {
                T::from(self) $o rhs.0
            }
        }

        impl $Op<Atom> for Atom{
            type Output = Atom;
            fn $op(self, rhs: Atom) -> Atom{
                match (self, rhs){
                    (One, One) => One,
                    _ => Zero
                }
            }
        }

    )*};
}

impl_node_val_ops!(add:Add+, mul:Mul*);
