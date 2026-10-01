//! The module traits: `Module` for a component whose output depends only on its
//! input, `ModuleT` for one that also depends on train/eval mode (dropout). Forward passes
//! panic on a bad shape, as the tensor core's methods do; constructors return `Result`.

use crate::tensor::Tensor;

/// A component with one tensor input and one tensor output.
pub trait Module {
    fn forward(&self, xs: &Tensor) -> Tensor;
}

/// A component whose forward pass depends on train (true) or eval (false) mode.
pub trait ModuleT {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor;
}

impl<M: Module> ModuleT for M {
    fn forward_t(&self, xs: &Tensor, _train: bool) -> Tensor {
        self.forward(xs)
    }
}

/// Any function of one tensor is a module.
impl<F: Fn(&Tensor) -> Tensor> Module for F {
    fn forward(&self, xs: &Tensor) -> Tensor {
        self(xs)
    }
}
