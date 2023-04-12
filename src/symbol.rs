use crate::{ops::Mul, prelude::*, Node};
use std::fmt::Debug;

#[derive(Clone)]
pub struct Symbol<N> {
    pub symbol: &'static str,
    pub node: N,
}

impl<'a, N: Differentiable<'a> + Debug> Debug for Symbol<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}={:?}", self.symbol, self.node)
    }
}

impl<N> Symbol<N> {
    pub fn new(node: N, symbol: &'static str) -> Self {
        Self { symbol, node }
    }
}

impl<'a, N: Differentiable<'a> + 'a> Differentiable<'a> for Symbol<N> {
    type Δ<D> = Mul<Atom, D>;
    type T = N::T;
    type Unit = N::Unit;

    fn eval(&self) -> Self::T {
        self.node.eval()
    }

    fn derivative<const LEN: usize, D>(&'a self, k: [&str; LEN], d: D) -> [Self::Δ<D>; LEN] {
        k.map(|k| Mul(if k == self.symbol { One } else { Zero }, d))
    }

    fn is_zero(&self) -> bool {
        false
    }
}

impl From<Atom> for () {
    fn from(_: Atom) {}
}

pub fn symbol(s: &'static str) -> Node<Symbol<()>> {
    Node(Symbol::new((), s))
}
