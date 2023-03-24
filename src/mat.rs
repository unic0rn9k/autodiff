use crate::Differentiable;

struct Float<T>(T);

pub struct MatrixNode<T, const R: usize, const C: usize, S: Storage<T, Const<R>, Const<C>>>(
    Matrix<T, Const<R>, Const<C>, S>,
);

impl<'a, T, R: Dim, C: Dim, S: Storage<T, R, C>> Differentiable<'a> for Matrix<T, R, C, S>
where
    Matrix<T, R, C, S>: Clone,
{
    type Δ = Atom;
    type T = NodeValue<Option<Matrix<T, R, C, S>>>;

    fn eval(&self) -> Self::T {
        NodeValue(Some(self.clone()))
    }

    fn derivative<const LEN: usize>(&'a self, _: &[&str; LEN]) -> [Self::Δ; LEN] {
        [Zero; LEN]
    }
}

impl<'a, T, const R: usize, const C: usize> Add<NodeValue<Option<SMatrix<T, R, C>>>>
    for NodeValue<Option<SMatrix<T, R, C>>>
{
    type Output = NodeValue<Option<SMatrix<T, R, C>>>;

    fn add(self, rhs: NodeValue<Option<SMatrix<T, R, C>>>) -> Self::Output {
        NodeValue(match (self.0, rhs.0) {
            (None, None) => None,
            (None, Some(r)) => Some(r),
            (Some(l), None) => Some(l),
            (Some(l), Some(r)) => Some(l + r),
        })
    }
}
