# autodiff
Autodiff graph, in rust, --for my final highschool programming project-- because I wan't to.

The idea is to be able to create optimized (compiled from expression trees) algebraic objects, that are differentiable.

# Features
- Statically sized matricies
- GPU support
- Blas and all that stuff

# TODO
- Safety, so you can't try to evaluate a node on a forrin graph
- Make miri stop being mad at matmul
- miri CI
