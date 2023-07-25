use std::marker::PhantomData;

use crate::{
    backend::*,
    graph::{self, Graph, Matrix, Node, NodeHash, NodeIndex, Scalar},
};

pub trait Op<T, B: Backend<T>> {
    type Output;
    type Compiled: 'static;

    // TODO: Need to return Result
    fn eval(
        this: &Self::Compiled,
        id: &NodeIndex<T, B, Self>,
        exe: &mut Graph<T, B>,
    ) -> Self::Output
    where
        Self: Sized;

    fn compile(self, graph: &mut graph::Graph<T, B>) -> graph::NodeIndex<T, B, Self>
    where
        Self: Sized;
}

#[derive(Clone, Copy)]
pub struct Add<L, R, T> {
    pub lhs: L,
    pub rhs: R,
    pub marker: PhantomData<T>,
}

impl<L, R, T: 'static> graph::Node for Add<L, R, T>
where
    L: graph::Node + 'static,
    R: graph::Node + 'static,
{
    fn hash(&self) -> graph::NodeHash {
        graph::NodeHash::collect(self, [&self.lhs, &self.rhs])
    }
}

macro_rules! impl_scalar_op {
    ($op:ident, $o:tt) => {
        impl<T: Copy, L, R> Op<T, CpuHeap> for $op<L, R, (Scalar, Scalar)>
        where
            T: std::ops::$op<T, Output = T>,
            L: Op<T, CpuHeap, Output = Scalar> + Node,
            R: Op<T, CpuHeap, Output = Scalar> + Node,
            $op<graph::NodeIndex<T, CpuHeap, L>, graph::NodeIndex<T, CpuHeap, R>, (Scalar, Scalar)>:
                'static,
        {
            type Output = Scalar;
            type Compiled = $op<
                graph::NodeIndex<T, CpuHeap, L>,
                graph::NodeIndex<T, CpuHeap, R>,
                (Scalar, Scalar),
            >;

            fn eval(this: &Self::Compiled, id: &NodeIndex<T, CpuHeap, Self>, exe: &mut Graph<T, CpuHeap>) -> Self::Output {
                let l = exe.eval(&this.lhs);
                let r = exe.eval(&this.rhs);

                exe.mut_data(id)[0] =
                    exe.get_scalar(&l).unwrap() $o exe.get_scalar(&r).unwrap();
                id.as_scalar(exe)
            }

            fn compile(self, graph: &mut graph::Graph<T, CpuHeap>) -> graph::NodeIndex<T, CpuHeap, Self>
            where
                Self: Sized,
            {
                let $op {
                    lhs,
                    rhs,
                    marker: PhantomData,
                } = &self;
                let hashes = [
                    NodeHash::collect(&self, [lhs, rhs]),
                    NodeHash::collect(&self, [rhs, lhs]),
                ];
                let $op {
                    lhs,
                    rhs,
                    marker: PhantomData,
                } = self;
                let lhs = lhs.compile(graph);
                let rhs = rhs.compile(graph);

                unsafe{
                    graph
                        .insert(
                            $op {
                                lhs,
                                rhs,
                                marker: PhantomData,
                            },
                            &hashes,
                            |b| b.static_alloc::<1>(),
                        )
                        .transmute()
                }
            }
        }
    };
}

#[derive(Clone, Copy)]
pub struct Mul<L, R, T> {
    pub lhs: L,
    pub rhs: R,
    pub marker: PhantomData<T>,
}

impl<L, R, T: 'static> graph::Node for Mul<L, R, T>
where
    L: graph::Node + 'static,
    R: graph::Node + 'static,
{
    fn hash(&self) -> graph::NodeHash {
        graph::NodeHash::collect(self, [&self.lhs, &self.rhs])
    }
}

impl_scalar_op!(Add, +);
impl_scalar_op!(Mul, *);

pub struct MatMul<L, R, T> {
    pub lhs: L,
    pub rhs: R,
    pub marker: PhantomData<T>,
}

impl<L, R, T: 'static> graph::Node for MatMul<L, R, T>
where
    L: graph::Node + 'static,
    R: graph::Node + 'static,
{
    fn hash(&self) -> graph::NodeHash {
        graph::NodeHash::collect(self, [&self.lhs, &self.rhs])
    }
}

impl<L, R, const M: usize, const N: usize, const K: usize> Op<f32, CpuHeap>
    for MatMul<L, R, (Matrix<M, N>, Matrix<N, K>)>
where
    L: Op<f32, CpuHeap, Output = Matrix<M, N>> + Node,
    R: Op<f32, CpuHeap, Output = Matrix<N, K>> + Node,
    MatMul<
        graph::NodeIndex<f32, CpuHeap, L>,
        graph::NodeIndex<f32, CpuHeap, R>,
        (Matrix<M, N>, Matrix<N, K>),
    >: 'static,
{
    type Output = Matrix<M, K>;
    type Compiled = MatMul<
        graph::NodeIndex<f32, CpuHeap, L>,
        graph::NodeIndex<f32, CpuHeap, R>,
        (Matrix<M, N>, Matrix<N, K>),
    >;

    fn eval(
        this: &Self::Compiled,
        id: &NodeIndex<f32, CpuHeap, Self>,
        exe: &mut Graph<f32, CpuHeap>,
    ) -> Self::Output {
        let l = exe.eval(&this.lhs);
        let r = exe.eval(&this.rhs);
        *exe.mut_data(id) = vec![0f32; M * K].into();
        let out = &mut exe.mut_data(id).as_mut()[0] as *mut f32;

        let l = exe.get_matrix(&l).unwrap();
        let r = exe.get_matrix(&r).unwrap();

        unsafe {
            matrixmultiply::sgemm(
                M,
                N,
                K,
                1.0,
                &l.as_ref()[0] as *const f32,
                N as isize,
                1,
                &r.as_ref()[0] as *const f32,
                K as isize,
                1,
                0.0,
                out,
                K as isize,
                1,
            )
        }

        id.as_matrix(exe)
    }

    fn compile(self, graph: &mut graph::Graph<f32, CpuHeap>) -> graph::NodeIndex<f32, CpuHeap, Self>
    where
        Self: Sized,
    {
        let MatMul {
            lhs,
            rhs,
            marker: PhantomData,
        } = &self;
        let hashes = [
            NodeHash::collect(&self, [lhs, rhs]),
            NodeHash::collect(&self, [rhs, lhs]),
        ];
        let MatMul {
            lhs,
            rhs,
            marker: PhantomData,
        } = self;
        let lhs = lhs.compile(graph);
        let rhs = rhs.compile(graph);
        unsafe {
            graph
                .insert(
                    MatMul {
                        lhs,
                        rhs,
                        marker: PhantomData,
                    },
                    &hashes,
                    |b| b.alloc(M * K),
                )
                .transmute()
        }
    }
}
