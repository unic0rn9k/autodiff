A library for automatic differentiation. [![Rust](https://github.com/unic0rn9k/autodiff/actions/workflows/rust.yml/badge.svg)](https://github.com/unic0rn9k/autodiff/actions/workflows/rust.yml)

## TODO
- [ ] separate `Mul` and `MatMul`
- [X] `.derivative()` should only take one gradient
- [ ] `exp` operator
- [ ] `sum` operator
- [ ] `div` operator

### softmax
sum = exp(x_0) + exp(x_1) + ... + exp(x_n)
softmax(x_n) = exp(x_n) / sum

softmax'(x_n) = softmax(x_n) * (1 - softmax(x_n))

f(x)/g(x) -> (f'(x)g(x) - f(x)g'(x)) / g(x)^2
exp(x) / sum -> (exp(x) * sum - exp(x) * exp(x)) / sum^2
