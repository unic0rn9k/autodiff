use petgraph::{graph::NodeIndex, graphmap::GraphMap, Directed};

trait IntoGraph {
    fn into_graph(self) -> (GraphMap<&'static str, &'static str, Directed>, NodeIndex);
}

impl<'a, L: 'a, R: 'a> IntoGraph for crate::ops::Mul<L, R>
where
    L: IntoGraph,
    R: IntoGraph,
{
    fn into_graph(self) -> (GraphMap<&'static str, &'static str, Directed>, NodeIndex) {
        let mut graph = GraphMap::default();
        let mut ret = graph.add_node("*");
        let mut lhs = self.0.into_graph();
        let mut rhs = self.1.into_graph();
        graph.extend_with_edges(lhs.0.all_edges());
        graph.extend_with_edges(rhs.0.all_edges());
        graph.add_edge(lhs.1., ret, "lhs");
        graph.add_edge(rhs.1, ret, "rhs");
        (graph, ret)
    }
}

// A function that takes a differentiable function and returns a graph
fn graph<'a, T: 'a, F: 'a>(f: F) -> (GraphMap<&'static str, &'static str, Directed>, NodeIndex)
where
    F: FnOnce() -> T,
    T: IntoGraph,
{
    f().into_graph()
}
