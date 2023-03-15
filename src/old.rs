#![feature(test)]

#[cfg(test)]
mod test;

use std::sync::Arc;

#[derive(PartialEq, Copy, Clone)]
struct NodeId(*const ());

trait DiffNode {
    type Δ;
    type T;

    fn eval(&self) -> Self::T;
    fn derivative(self: &Arc<Self>, k: &[NodeId]) -> Vec<Arc<Self::Δ>>;
    fn id(self: &Arc<Self>) -> NodeId {
        NodeId(Arc::as_ptr(self) as *const ())
    }
}

impl DiffNode for f32 {
    type Δ = f32;
    type T = f32;

    fn eval(&self) -> Self::T {
        *self
    }

    fn derivative(self: &Arc<Self>, k: &[NodeId]) -> Vec<Arc<Self::Δ>> {
        k.iter()
            .map(|k| Arc::new(if *k == self.id() { 1. } else { 0. }))
            .collect()
    }
}

struct Add<Lhs, Rhs>(Arc<Lhs>, Arc<Rhs>);

impl<Lhs: DiffNode<T = f32>, Rhs: DiffNode<T = f32>> DiffNode for Add<Lhs, Rhs> {
    type Δ = Add<Lhs::Δ, Rhs::Δ>;
    type T = f32;

    fn eval(&self) -> Self::T {
        self.0.eval() + self.1.eval()
    }

    fn derivative(self: &Arc<Self>, k: &[NodeId]) -> Vec<Arc<Self::Δ>> {
        self.0
            .derivative(k)
            .iter()
            .zip(self.1.derivative(k).iter())
            .map(|(dx, dy)| Arc::new(Add(dx.clone(), dy.clone())))
            .collect()
    }
}

struct Mul<Lhs, Rhs>(Arc<Lhs>, Arc<Rhs>);

impl<Lhs: DiffNode<T = f32>, Rhs: DiffNode<T = f32>> DiffNode for Mul<Lhs, Rhs> {
    type T = f32;
    type Δ = Add<Mul<Lhs, Rhs::Δ>, Mul<Rhs, Lhs::Δ>>;

    fn eval(&self) -> Self::T {
        self.0.eval() * self.1.eval()
    }

    fn derivative(self: &Arc<Self>, k: &[NodeId]) -> Vec<Arc<Self::Δ>> {
        let dx = self.0.derivative(k);
        let dy = self.1.derivative(k);
        let Mul(x, y) = self.as_ref();
        dx.iter()
            .zip(dy.iter())
            .map(|(dx, dy)| {
                Arc::new(Add(
                    Arc::new(Mul(x.clone(), dy.clone())),
                    Arc::new(Mul(y.clone(), dx.clone())),
                ))
            })
            .collect()
    }
}

macro_rules! impl_debug {
    ($op: tt $name: ident) => {
        impl<Lhs: std::fmt::Debug, Rhs: std::fmt::Debug> std::fmt::Debug for $name<Lhs, Rhs> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "({:?} {} {:?})", self.0, stringify!($op), self.1)
            }
        }
    };
}

impl_debug!(*Mul);
impl_debug!(+Add);
