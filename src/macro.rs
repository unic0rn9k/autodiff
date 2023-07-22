macro_rules! def_op {
    ($name:ident{$($field:ident:$arg:ident),*} ($self: ident, $exe: ident) $backend:ty => $eval:expr $(,$($etc:tt)*)?) => {
        pub struct $name<$($arg),*>{$(pub $field: $arg),*}

        def_op!(=> $name{$($field:$arg),*} ($self, $exe) $backend => $eval $($($etc)*)?);
    };
    (=> $name:ident{$($field:ident:$arg:ident),*} ($self: ident, $exe: ident) $backend:ty => $eval:expr $(,$($etc:tt)*)?) => {
        impl<$($arg: Node + 'static),*> Node for $name<$($arg),*> {
            fn hash(&self) -> graph::NodeHash {
                graph::NodeHash::collect(self, [$(&self.$field),*])
            }
        }

        impl<T, $($arg),*> Op<T, $backend> for $name<$($arg),*>
        where
            $($arg: Op<T, $backend> + Node,)*
            T: std::ops::$name + 'static,
        {
            type Output = <T as std::ops::$name>::Output;

            fn eval(&self, $exe: &$backend) -> Self::Output {
                let $self = self;
                $eval
            }

            fn compile(
                self,
                graph: &mut graph::Graph<T, $backend>,
            ) -> graph::CompiledNode<T, CpuArc, impl Op<T, $backend> + 'static>
            where
                Self: Sized,
            {
                $(let $field = self.$field.compile(graph);)*
                graph.insert($name{$($field),*})
            }
        }

        def_op!(=> $name{$($field:$arg),*} $($($etc)*)?);
    };
    (=> $name:ident{$($field:ident:$arg:ident),*}) => {};
}

def_op! {
    Add{lhs:L, rhs:R} (this, exe)
        CpuArc => this.lhs.eval(exe) + this.rhs.eval(exe),
        // GpuArc where T: ... => todo!(),
}
