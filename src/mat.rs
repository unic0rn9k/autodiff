use crate::ops::Transpose;
use crate::value::Atom;
use crate::{prelude::*, value, Node};
use nalgebra::{DMatrix, Dyn, OMatrix};
use std::fmt::Debug;
use std::ops::*;

#[derive(Clone)]
pub struct MatrixNode<T>(pub Option<OMatrix<T, Dyn, Dyn>>)
where
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<T, nalgebra::Dyn, nalgebra::Dyn>;

impl<T> MatrixNode<T>
where
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<T, nalgebra::Dyn, nalgebra::Dyn>,
{
    pub fn shape(&self) -> Option<(usize, usize)> {
        self.0.as_ref().map(|m| (m.nrows(), m.ncols()))
    }
}

impl<T> Debug for MatrixNode<T>
where
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<T, nalgebra::Dyn, nalgebra::Dyn>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some((r, c)) = self.shape() {
            write!(f, " [{r}x{c}] ")
        } else {
            write!(f, "Zero")
        }
    }
}

//impl<'a, T: Copy + PartialEq + std::fmt::Debug + 'static> Differentiable<'a>
//    for OMatrix<T, Dyn, Dyn>
//{
//    type Δ = Atom;
//    type T = MatrixNode<T>;
//
//    fn eval(&self) -> Self::T {
//        MatrixNode(Some(self.clone()))
//    }
//
//    fn derivative<const LEN: usize>(&'a self, _: &[&str; LEN]) -> [Self::Δ; LEN] {
//        [Zero; LEN]
//    }
//}

impl<T: nalgebra::Scalar + crate::value::Scalar> PartialEq for MatrixNode<T>
where
    OMatrix<T, Dyn, Dyn>: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        match (self.0.as_ref(), other.0.as_ref()) {
            (Some(a), Some(b)) => {
                if a.nrows() != b.nrows() || a.ncols() != b.ncols() {
                    return false;
                }
                for i in 0..a.nrows() {
                    for j in 0..a.ncols() {
                        if a[(i, j)] != b[(i, j)] {
                            return false;
                        }
                    }
                }
                true
            }
            (None, None) => true,
            _ => false,
        }
    }
}

impl<'a, T: Copy + PartialEq + std::fmt::Debug + 'static> Differentiable<'a> for MatrixNode<T> {
    type Δ<D> = Atom;
    type T = MatrixNode<T>;

    fn eval(&self) -> Self::T {
        self.clone()
    }

    fn derivative<const LEN: usize, D>(&'a self, _: [(&str, D); LEN]) -> [Self::Δ<D>; LEN] {
        [Zero; LEN]
    }
}

impl<T> Add<MatrixNode<T>> for MatrixNode<T>
where
    OMatrix<T, Dyn, Dyn>: Add<OMatrix<T, Dyn, Dyn>, Output = OMatrix<T, Dyn, Dyn>>,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<T, nalgebra::Dyn, nalgebra::Dyn>,
{
    type Output = MatrixNode<T>;

    fn add(self, rhs: MatrixNode<T>) -> Self::Output {
        MatrixNode(match (self.0, rhs.0) {
            (None, None) => None,
            (None, Some(r)) => Some(r),
            (Some(l), None) => Some(l),
            (Some(l), Some(r)) => Some(l + r),
        })
    }
}

impl<T> Mul<MatrixNode<T>> for MatrixNode<T>
where
    OMatrix<T, Dyn, Dyn>: Mul<OMatrix<T, Dyn, Dyn>, Output = OMatrix<T, Dyn, Dyn>>,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<T, nalgebra::Dyn, nalgebra::Dyn>,
{
    type Output = MatrixNode<T>;

    fn mul(self, rhs: MatrixNode<T>) -> Self::Output {
        let (l, r) = (self, rhs);
        let err = format!("{l:?} cannot be multiplied by {r:?}");

        MatrixNode(match (l.0, r.0) {
            (Some(l), Some(r)) => {
                assert_eq!(l.ncols(), r.nrows(), "{err}");
                Some(l * r)
            }
            _ => None,
        })
    }
}

impl<T: nalgebra::Scalar + crate::value::Scalar> Mul<Atom> for MatrixNode<T>
where
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<T, nalgebra::Dyn, nalgebra::Dyn>,
{
    type Output = MatrixNode<T>;

    fn mul(self, rhs: Atom) -> Self::Output {
        if rhs == value::Zero {
            return Self(None);
        }
        self
    }
}

impl<T: nalgebra::Scalar + crate::value::Scalar> Mul<MatrixNode<T>> for Atom
where
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<T, nalgebra::Dyn, nalgebra::Dyn>,
{
    type Output = MatrixNode<T>;

    fn mul(self, rhs: MatrixNode<T>) -> Self::Output {
        rhs * self
    }
}

impl<T: nalgebra::Scalar + crate::value::Scalar + Add<T, Output = T>> Add<Atom> for MatrixNode<T> {
    type Output = MatrixNode<T>;

    fn add(self, rhs: Atom) -> Self::Output {
        Self(self.0.map(|m| m.map(|n| n + T::from(rhs))))
    }
}

impl<T: nalgebra::Scalar + crate::value::Scalar + Add<T, Output = T>> Add<MatrixNode<T>> for Atom {
    type Output = MatrixNode<T>;

    fn add(self, rhs: MatrixNode<T>) -> Self::Output {
        rhs + self
    }
}

pub fn mat<T>(m: OMatrix<T, Dyn, Dyn>) -> MatrixNode<T>
where
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<T, nalgebra::Dyn, nalgebra::Dyn>,
{
    MatrixNode(Some(m))
}

#[test]
fn gradient_decent() {
    let x = mat(DMatrix::<f32>::new_random(3, 1)).symbol("input");

    let w1 = mat(DMatrix::<f32>::new_random(2, 3)).symbol("w1");
    let b1 = mat(DMatrix::<f32>::new_random(2, 1)).symbol("b1");

    let w2 = mat(DMatrix::<f32>::new_random(4, 2)).symbol("w2");
    let b2 = mat(DMatrix::<f32>::new_random(4, 1)).symbol("b2");

    let l1 = &w1 * &x + &b1;
    let l2 = &w2 * &l1 + &b2;
    let [dw1, dw2] = l2.derivative([("w1", One), ("w2", One)]);

    l2.eval();

    println!("dw1 = {dw1:?}");
    println!();
    println!("dw2 = {dw2:?}");
    println!();

    assert_eq!(dw2.eval(), (&l1).transpose().eval());
    assert_eq!(dw1.eval(), (Node(Transpose(w2)) * x).transpose().eval());
}
