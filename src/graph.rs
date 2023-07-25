use crate::{backend::Backend, error::Result, expr::Expr, ops::Op};
use std::{
    any::Any,
    collections::HashMap,
    fmt::{Debug, Formatter},
    marker::PhantomData,
    mem::transmute,
    sync::Arc,
};

/// A Node can be inserted into a Graph.
pub trait Node {
    /// Returns a Hash of self, for checking for duplicates.
    fn hash(&self) -> NodeHash;
}

/// An unsafe way of representing the Hash of a Node.
#[derive(Hash, Eq, PartialEq, Debug, Clone)]
pub struct NodeHash {
    type_id: usize,
    data_id: Vec<usize>,
}

/// Return a unique identifier for a type, represented as a usize.
/// (its just the vtable pointer of the fat pointer)
fn type_id(fatty: &dyn Any) -> usize {
    unsafe { transmute::<_, (usize, usize)>(fatty).1 }
}

impl NodeHash {
    /// A helper function to collect the hashes of a Node and its subnodes.
    pub fn collect<const N: usize>(root: &dyn Any, sub: [&dyn Node; N]) -> Self {
        Self {
            type_id: type_id(root),
            data_id: sub.iter().flat_map(|x| x.hash().data_id).collect(),
        }
    }
}

/// A container for the original data of a Node,
/// with the type information obfuscated.
/// Also the `data` field points to the slot in the Graph's data,
/// containing the output value of the node.
pub struct CompiledNode<T, B: Backend<T>> {
    inner: Arc<dyn Any>,
    data_id: usize,
    name: &'static str,
    marker: PhantomData<(T, B)>,
}

impl<T, B: Backend<T>> CompiledNode<T, B> {
    //fn typed<N: Node>(&self) -> &CompiledNode<T, B, N> {
    //    if !self.inner.is::<N>() {
    //        panic!("Node type mismatch")
    //    }
    //    //unsafe { transmute(self) }
    //    todo!()
    //}
}

/// Points to the position of the CompiledNode in its graph.
#[derive(Clone)]
pub struct NodeIndex<T, B: Backend<T>, N> {
    pub id: usize, // TODO: Remove pub
    hash: NodeHash,
    marker: std::marker::PhantomData<(T, B, N)>,
}

impl<T, B: Backend<T>, N> NodeIndex<T, B, N> {
    /// Transmute the type of the source node.
    /// Roughly corresponds to transmuting a &T to a &U.
    /// Use `transmute_ref` if self is a ref.
    ///
    /// # Safety
    /// Very unsafe. Make sure you know what you're doing.
    /// Read about `std::mem::transmute` for more information.
    pub unsafe fn transmute<Dst>(self) -> NodeIndex<T, B, Dst> {
        NodeIndex {
            id: self.id,
            hash: self.hash,
            marker: std::marker::PhantomData,
        }
    }

    /// Transmute the type of the source node.
    /// Roughly corresponds to transmuting a &T to a &U.
    /// Use `transmute` to consume self.
    ///
    /// # Safety
    /// Very unsafe. Make sure you know what you're doing.
    /// Read about `std::mem::transmute` for more information.
    pub unsafe fn transmute_ref<Dst>(&self) -> &NodeIndex<T, B, Dst> {
        transmute(self)
    }

    /// Returns the node's output as a Scalar.
    pub fn as_scalar(&self, graph: &Graph<T, B>) -> Scalar
    where
        N: Op<T, B, Output = Scalar>,
    {
        // TODO: What to do about this symbol?
        Scalar(graph.nodes[self.id].data_id, "")
    }

    /// Returns the node's output as a Matrix.
    pub fn as_matrix<const R: usize, const C: usize>(&self, graph: &Graph<T, B>) -> Matrix<R, C>
    where
        N: Op<T, B, Output = Matrix<R, C>>,
    {
        Matrix(
            graph.nodes[self.id].data_id,
            MatrixMeta {
                trans: false,
                leading_dim: R,
            },
        )
    }
}

impl<T, B: Backend<T>, N: Op<T, B> + 'static> Op<T, B> for NodeIndex<T, B, N> {
    type Output = N::Output;
    type Compiled = N;

    fn eval(_: &N, _: &NodeIndex<T, B, Self>, _: &mut Graph<T, B>) -> Self::Output {
        unimplemented!()
    }

    fn compile(self, _: &mut Graph<T, B>) -> NodeIndex<T, B, Self>
    where
        Self: Sized,
    {
        panic!("Node already compiled")
    }
}

impl<T, B: Backend<T>, N> Node for NodeIndex<T, B, N> {
    fn hash(&self) -> NodeHash {
        self.hash.clone()
    }
}
impl<'a, T, B: Backend<T>, N> Node for &'a NodeIndex<T, B, N> {
    fn hash(&self) -> NodeHash {
        self.hash.clone()
    }
}

/// A Graph is a container for Nodes. A graph can contain multiple separate expressions.
pub struct Graph<T, B: Backend<T>> {
    cache: HashMap<NodeHash, usize>,
    nodes: Vec<CompiledNode<T, B>>,
    pub backend: B,
    pub data: Vec<B::DevicePtr>,
}

impl<T, B: Backend<T>> Graph<T, B> {
    /// Insert a node into the graph.
    /// `hashes` should contain all nodes permutations that will evaluate to the same value as `node`.
    /// fx a+b and b+a should both be included for the node corresponding to a+b.
    /// It might be possible to build this functionality directly into the hashing system.
    pub fn insert<N: 'static + Op<T, B>>(
        &mut self,
        node: N,
        hashes: &[NodeHash],
        alloc: impl Fn(&B) -> Result<B::DevicePtr>,
    ) -> NodeIndex<T, B, N> {
        let id = self.nodes.len();

        let hash = hashes
            .iter()
            .find(|hash| self.cache.contains_key(hash))
            .unwrap_or(&hashes[0])
            .clone();

        let id = *self.cache.entry(hash.clone()).or_insert_with(|| {
            let data_id = self.data.len();
            self.data.push(alloc(&self.backend).unwrap());

            self.nodes.push(CompiledNode {
                inner: Arc::new(node),
                data_id,
                name: std::any::type_name::<N>(),
                marker: PhantomData,
            });
            id
        });
        NodeIndex {
            id,
            hash,
            marker: std::marker::PhantomData,
        }
    }

    /// Create a new graph with the given backend.
    pub fn new(backend: B) -> Self {
        Self {
            cache: HashMap::new(),
            nodes: Vec::new(),
            backend,
            data: vec![],
        }
    }

    /// Allocate, insert and return the index, for a new scalar node, containing the value `value`.
    pub fn scalar(&mut self, value: T, symbol: &'static str) -> Result<Expr<Scalar, T, B>> {
        let n = self.data.len();
        self.data.push(self.backend.htod(vec![value])?);
        Ok(Expr(Scalar(n, symbol), PhantomData))
    }

    pub fn matrix<const R: usize, const C: usize>(
        &mut self,
        value: Vec<T>,
    ) -> Result<Expr<Matrix<R, C>, T, B>> {
        assert_eq!(R * C, value.len());
        let n = self.data.len();
        self.data.push(self.backend.htod(value)?);
        Ok(Expr(
            Matrix(
                n,
                MatrixMeta {
                    trans: false,
                    leading_dim: 0,
                },
            ),
            PhantomData,
        ))
    }

    /// Evaluate the node at `node_src`.
    pub fn eval<N: Node + Op<T, B>>(&mut self, node_src: &NodeIndex<T, B, N>) -> N::Output {
        let node = self.nodes[node_src.id].inner.clone();
        //let storage = &mut self.nodes[node_src.id].data;
        let node: &N::Compiled = node.downcast_ref().unwrap_or_else(|| {
            panic!(
                "{} != {}\non node{} in\n{self:?}",
                std::any::type_name::<N::Compiled>(),
                self.nodes[node_src.id].name,
                node_src.id,
            )
        });

        #[cfg(not(debug_assertions))]
        let node = unsafe {
            transmute::<_, (&N::Compiled, &())>(self.nodes[node_src.id].inner.as_ref()).0
        };

        N::eval(node, node_src, self)
    }

    /// Get a reference for the data of `node`.
    pub fn data<N>(&self, node: &NodeIndex<T, B, N>) -> &B::DevicePtr
    where
        N: Node,
    {
        &self.data[self.nodes[node.id].data_id]
    }

    /// Get a mutable reference for the data of `node`.
    pub fn mut_data<N>(&mut self, node: &NodeIndex<T, B, N>) -> &mut B::DevicePtr
    where
        N: Node,
    {
        &mut self.data[self.nodes[node.id].data_id]
    }

    /// Get the value of a Scalar.
    pub fn get_scalar(&self, s: &Scalar) -> Result<T>
    where
        T: Copy,
    {
        Ok(self.backend.dtoh(&self.data[s.0])?[0])
    }

    /// Get the value of a Matrix.
    pub fn get_matrix<const R: usize, const C: usize>(&self, m: &Matrix<R, C>) -> Result<Box<[T]>>
    where
        T: Copy,
    {
        self.backend.dtoh(&self.data[m.0])
    }
}

impl<T, B: Backend<T>> Debug for Graph<T, B> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        for (n, node) in self.nodes.iter().enumerate() {
            writeln!(f, "{n}: {}", node.name)?;
        }
        Ok(())
    }
}

/// A node that points to a single scalar value.
/// A Scalar just evaluates to itself.
/// In the implementation of other operations,
/// the actual value of the scalar can be fetched, if needed.
#[derive(Clone, Copy)]
pub struct Scalar(usize, &'static str);

impl Node for Scalar {
    fn hash(&self) -> NodeHash {
        NodeHash {
            type_id: type_id(self),
            data_id: vec![self.0],
        }
    }
}

impl<T: 'static, B: Backend<T> + 'static> Op<T, B> for Scalar {
    type Output = Self;
    type Compiled = Self;

    fn eval(this: &Self, _: &NodeIndex<T, B, Self>, _: &mut Graph<T, B>) -> Self::Output {
        *this
    }

    fn compile(self, graph: &mut Graph<T, B>) -> NodeIndex<T, B, Self>
    where
        Self: Sized,
    {
        let hash = self.hash();
        graph.insert(self, &[hash], |b| b.static_alloc::<1>())
    }
}

// Any additional information needed to be associated with a matrix,
// besides the index of its `CompiledNode`.
#[derive(Clone, Copy)]
pub struct MatrixMeta {
    pub trans: bool,
    pub leading_dim: usize,
}

/// A node that points to a matrix.
/// Note I don't actually prioritize static allocation,
/// the const generics are just for compile time assertions.
#[derive(Clone, Copy)]
pub struct Matrix<const R: usize, const C: usize>(usize, MatrixMeta);

impl<const R: usize, const C: usize> Node for Matrix<R, C> {
    fn hash(&self) -> NodeHash {
        NodeHash {
            type_id: type_id(self),
            data_id: vec![self.0],
        }
    }
}

impl<T: 'static, B: Backend<T> + 'static, const R: usize, const C: usize> Op<T, B>
    for Matrix<R, C>
{
    type Output = Self;
    type Compiled = Self;

    fn eval(this: &Self, _: &NodeIndex<T, B, Self>, _: &mut Graph<T, B>) -> Self::Output {
        *this
    }

    fn compile(self, graph: &mut Graph<T, B>) -> NodeIndex<T, B, Self>
    where
        Self: Sized,
    {
        let hash = self.hash();
        graph.insert(self, &[hash], |b| b.alloc(R * C))
    }
}
