use crate::primitive_ops::*;
use crate::Differentiable;
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

    fn derivative<const LEN: usize, D>(&'a self, _: [&str; LEN], _d: D) -> [Self::Δ<D>; LEN] {
        [Zero; LEN]
    }

    fn is_zero(&self) -> bool {
        match self {
            Zero => true,
            One => false,
        }
    }
}

pub trait Scalar: From<Atom> {}

macro_rules! impl_scalar {
    ($($t: ident)*) => {$(
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

                fn derivative<const LEN: usize, D>(&'a self, _: [&str; LEN], _d:D) -> [Self::Δ< D>; LEN] {
                    [Zero; LEN]
                }

                fn is_zero(&self) -> bool{
                    false
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
    ($($op:ident:$Op:ident)*) => {$(
        impl<L: $Op<R>, R> $Op<NodeValue<R>> for NodeValue<L> {
            type Output = NodeValue<<L as $Op<R>>::Output>;

            fn $op(self, rhs: NodeValue<R>) -> Self::Output {
                NodeValue(self.0.$op(rhs.0))
            }
        }

        impl<T: Scalar + $Op<Output=T>> $Op<Atom> for NodeValue<T>{
            type Output = NodeValue<T>;

            fn $op(self, rhs: Atom) -> Self::Output {
                NodeValue(self.0.$op(T::from(rhs)))
            }
        }

        impl<T: Scalar + $Op<Output=T>> $Op<NodeValue<T>> for Atom{
            type Output = NodeValue<T>;

            fn $op(self, rhs: NodeValue<T>) -> Self::Output {
                NodeValue(T::from(self).$op(rhs.0))
            }
        }
    )*};
}

impl_node_val_ops!(add:Add mul:Mul sub:Sub div:Div elem_mul:ElemMul);

impl<T: Neg> Neg for NodeValue<T> {
    type Output = NodeValue<<T as Neg>::Output>;

    fn neg(self) -> Self::Output {
        NodeValue(-self.0)
    }
}

impl<T: Exp> Exp for NodeValue<T> {
    type Output = NodeValue<<T as Exp>::Output>;

    fn exp(self) -> Self::Output {
        NodeValue(self.0.exp())
    }
}

impl Mul<Atom> for Atom {
    type Output = Atom;

    fn mul(self, rhs: Atom) -> Self::Output {
        match (self, rhs) {
            (One, One) => One,
            _ => Zero,
        }
    }
}

impl Add<Atom> for Atom {
    type Output = Atom;

    fn add(self, rhs: Atom) -> Self::Output {
        match (self, rhs) {
            (Zero, _) => rhs,
            (_, Zero) => self,
            (One, One) => panic!("Atom overflow"),
        }
    }
}

impl MulAssign for Atom {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}
