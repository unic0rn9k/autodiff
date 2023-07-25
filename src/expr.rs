use crate::{
    backend::Backend,
    graph::{Node, NodeIndex},
    ops::Op,
};

/// Expr is a wrapper around an Op that allows for operator overloading.
/// `E` is the underlying Op, `T` is the scalar type of the backend, and `B` is the backend.
#[derive(Debug)]
pub struct Expr<E, T, B>(pub E, pub std::marker::PhantomData<(T, B)>);
impl<E: Clone, T, B> Clone for Expr<E, T, B> {
    fn clone(&self) -> Self {
        Self(self.0.clone(), self.1)
    }
}
impl<E: Copy, T, B> Copy for Expr<E, T, B> {}

impl<T, B: Backend<T>, E: Op<T, B>> Op<T, B> for Expr<E, T, B> {
    type Output = E::Output;
    type Compiled = E::Compiled;

    fn eval(
        this: &Self::Compiled,
        id: &NodeIndex<T, B, Self>,
        exe: &mut crate::graph::Graph<T, B>,
    ) -> Self::Output {
        E::eval(this, unsafe { id.transmute_ref() }, exe)
    }

    fn compile(self, graph: &mut crate::graph::Graph<T, B>) -> crate::graph::NodeIndex<T, B, Self>
    where
        Self: Sized,
    {
        unsafe { self.0.compile(graph).transmute() }
    }
}

impl<E: Node, T, B> Node for Expr<E, T, B> {
    fn hash(&self) -> crate::graph::NodeHash {
        self.0.hash()
    }
}

macro_rules! impl_op {
    ($($Op:ident:$op:ident),*) => {
        $(
            impl<L: Op<T,B>, R: Op<T,B>, T, B: Backend<T>> std::ops::$Op<Expr<R,T,B>> for Expr<L,T,B>
            {
                type Output = Expr<crate::ops::$Op<L, R, (L::Output, R::Output)>, T,B>;

                fn $op(self, rhs: Expr<R, T,B>) -> Self::Output {
                    Expr(crate::ops::$Op {
                        lhs: self.0,
                        rhs: rhs.0,
                        marker: std::marker::PhantomData
                    }, std::marker::PhantomData)
                }
            }
        )*
    };
}

#[allow(type_alias_bounds)]
type MM<E: Op<T, B>, E2: Op<T, B>, T, B: Backend<T>> =
    Expr<crate::ops::MatMul<E, E2, (E::Output, E2::Output)>, T, B>;

impl_op!(Add:add, Mul:mul);

impl<E, T, B: Backend<T>> Expr<E, T, B> {
    pub fn matmul<E2>(self, rhs: Expr<E2, T, B>) -> MM<E, E2, T, B>
    where
        E: Op<T, B>,
        E2: Op<T, B>,
    {
        Expr(
            crate::ops::MatMul {
                lhs: self.0,
                rhs: rhs.0,
                marker: std::marker::PhantomData,
            },
            std::marker::PhantomData,
        )
    }
}
