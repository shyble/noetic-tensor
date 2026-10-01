//! Reverse-mode autodiff. The traversal is burn-autodiff 0.21's (a valid topological
//! order; changing it would only reorder gradient sums).
//!
//! Every float operation gives its output an `order` of 1 + the largest order of its float
//! inputs (tracked or not); fresh tensors, int/bool conversions and `detach` start at 0, and a
//! `require_grad` leaf is order 0. A tracked output gets a node that lists its tracked inputs.
//! `backward` walks the graph exactly as burn's `BreadthFirstSearch` does (a stack of parent
//! ids, visited on first pop), groups the nodes by order, runs the groups from the deepest down
//! and, inside a group, in visit order. A node's gradient is the sum of its consumers'
//! contributions in arrival order (`new + old`), which is burn's `Gradients::register`.

use super::Tensor;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

static NEXT_ID: AtomicU64 = AtomicU64::new(0);

pub(crate) fn next_id() -> u64 {
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}

/// The gradients of one operation's inputs from its output's gradient; `needs[i]` is false for
/// inputs that are not tracked (their slot may be None).
pub(crate) type BackwardFn = Box<dyn Fn(Tensor, &[bool]) -> Vec<Option<Tensor>> + Send + Sync>;

pub(crate) struct Node {
    pub(crate) id: u64,
    pub(crate) order: usize,
    /// The operation's float inputs, None where an input is not tracked.
    pub(crate) inputs: Vec<Option<Arc<Node>>>,
    /// None for a leaf (a `require_grad` tensor).
    pub(crate) backward: Option<BackwardFn>,
}

/// Drop a graph iteratively: a long chain of nodes would otherwise recurse once per node and
/// could overflow the stack.
impl Drop for Node {
    fn drop(&mut self) {
        let mut stack: Vec<Arc<Node>> = self.inputs.drain(..).flatten().collect();
        while let Some(n) = stack.pop() {
            if let Some(mut inner) = Arc::into_inner(n) {
                stack.extend(inner.inputs.drain(..).flatten());
            }
        }
    }
}

impl Node {
    pub(crate) fn leaf() -> Arc<Node> {
        Arc::new(Node { id: next_id(), order: 0, inputs: vec![], backward: None })
    }

    fn parent_ids(&self) -> impl Iterator<Item = u64> + '_ {
        self.inputs.iter().flatten().map(|n| n.id)
    }
}

/// Gradients of the tracked leaves, from one `backward` call.
pub struct Gradients {
    map: HashMap<u64, Tensor>,
}

impl Gradients {
    /// The gradient of a `require_grad` tensor, if it took part in the graph.
    pub fn get(&self, t: &Tensor) -> Option<Tensor> {
        t.node.as_ref().and_then(|n| self.map.get(&n.id)).cloned()
    }

    /// Remove and return a leaf's gradient.
    pub fn remove(&mut self, t: &Tensor) -> Option<Tensor> {
        t.node.as_ref().and_then(|n| self.map.remove(&n.id))
    }
}

/// Build an operation's output: order from all inputs, a node when any input is tracked.
/// `record` for an output of any dtype.
pub(crate) fn record_storage(storage: super::storage::Storage, shape: Vec<usize>, inputs: &[&Tensor], backward: impl FnOnce() -> BackwardFn) -> Tensor {
    record_with_output(storage, shape, inputs, |_| backward())
}

/// `record_storage` for an operation whose backward reads its own output (the standard
/// formulations save the result: sigmoid, exp, sqrt, recip, softmax, log_softmax, logsumexp).
/// `backward` receives the output, untracked, sharing its storage.
pub(crate) fn record_with_output(storage: super::storage::Storage, shape: Vec<usize>, inputs: &[&Tensor], backward: impl FnOnce(Tensor) -> BackwardFn) -> Tensor {
    let order = inputs.iter().map(|t| t.order).max().unwrap_or(0) + 1;
    let tracked = inputs.iter().any(|t| t.node.is_some());
    let device = inputs.first().map(|t| t.device).unwrap_or_default();
    // Every op branches to its GPU kernel before reading host values, so its output already
    // lives on the device. A host result for a GPU tensor means an op without a
    // GPU branch: a debug build stops; a release build uploads it and counts a round trip.
    debug_assert!(super::gpu::agrees(&storage, device), "an op computed on the host for a {device:?} tensor");
    let storage = if super::gpu::agrees(&storage, device) {
        Arc::new(storage)
    } else {
        super::gpu::count_round_trip();
        super::error::ok(super::gpu::place(&Arc::new(storage), device))
    };
    let mut out = Tensor { storage, layout: super::layout::Layout::contiguous(shape), order, node: None, device };
    if tracked {
        let saved = Tensor { storage: out.storage.clone(), layout: out.layout.clone(), order: 0, node: None, device };
        out.node = Some(Arc::new(Node { id: next_id(), order, inputs: inputs.iter().map(|t| t.node.clone()).collect(), backward: Some(backward(saved)) }));
    }
    out
}

/// Attach an operation's node to an output computed from untracked copies of `inputs` (a
/// composite forward with its own backward: softmax, log_softmax, logsumexp). `backward`
/// receives the output, untracked.
pub(crate) fn attach(out: Tensor, inputs: &[&Tensor], backward: impl FnOnce(Tensor) -> BackwardFn) -> Tensor {
    debug_assert!(out.node.is_none(), "attach takes an untracked output");
    let order = inputs.iter().map(|t| t.order).max().unwrap_or(0) + 1;
    let tracked = inputs.iter().any(|t| t.node.is_some());
    let node = tracked.then(|| {
        let saved = Tensor { storage: out.storage.clone(), layout: out.layout.clone(), order: 0, node: None, device: out.device };
        Arc::new(Node { id: next_id(), order, inputs: inputs.iter().map(|t| t.node.clone()).collect(), backward: Some(backward(saved)) })
    });
    Tensor { order, node, ..out }
}

/// A view's output: the input's storage under a new layout (no copy); order and node as
/// `record` gives them, so the graph is the copying op's graph.
pub(crate) fn record_view(input: &Tensor, layout: super::layout::Layout, backward: impl FnOnce() -> BackwardFn) -> Tensor {
    let order = input.order + 1;
    let node = input.node.as_ref().map(|_| Arc::new(Node { id: next_id(), order, inputs: vec![input.node.clone()], backward: Some(backward()) }));
    Tensor { storage: input.storage.clone(), layout, order, node, device: input.device }
}

pub(crate) fn run_backward(root: &Tensor) -> Gradients {
    super::error::ok(super::infer::backward_root(&root.layout.shape, root.node.is_some()));
    let root_node = root.node.clone().expect("tracked");
    // Every node reachable from the root, by id (burn keeps them in the server's step map).
    let mut steps: HashMap<u64, Arc<Node>> = HashMap::new();
    let mut stack = vec![root_node.clone()];
    while let Some(n) = stack.pop() {
        if steps.contains_key(&n.id) {
            continue;
        }
        for p in n.inputs.iter().flatten() {
            stack.push(p.clone());
        }
        steps.insert(n.id, n);
    }
    // burn's BreadthFirstSearch::traverse, filling the tape by depth.
    let mut tape: Vec<Vec<Arc<Node>>> = (0..root_node.order + 1).map(|_| Vec::new()).collect();
    let mut visited = HashSet::new();
    visited.insert(root_node.id);
    let mut parents: Vec<u64> = root_node.parent_ids().collect();
    steps.remove(&root_node.id);
    tape[root_node.order].push(root_node.clone());
    while let Some(id) = parents.pop() {
        let Some(step) = steps.remove(&id) else { continue };
        if visited.contains(&step.id) {
            continue;
        }
        visited.insert(step.id);
        for p in step.parent_ids() {
            if !visited.contains(&p) {
                parents.push(p);
            }
        }
        if let Some(level) = tape.get_mut(step.order) {
            level.push(step);
        }
    }
    let mut grads: HashMap<u64, Tensor> = HashMap::new();
    grads.insert(root_node.id, Tensor::full_like(root.layout.shape.clone(), 1.0, root));
    for level in tape.into_iter().rev() {
        for node in level {
            let Some(backward) = &node.backward else { continue }; // a leaf: its gradient stays
            let grad = grads.remove(&node.id).expect("a node's gradient is registered before its step runs");
            let needs: Vec<bool> = node.inputs.iter().map(|i| i.is_some()).collect();
            let outs = backward(grad, &needs);
            for (input, g) in node.inputs.iter().zip(outs) {
                let Some(input) = input else { continue };
                let g = g.expect("a tracked input receives a gradient");
                let g = match grads.remove(&input.id) {
                    Some(old) => g.add_raw(&old),
                    None => g,
                };
                grads.insert(input.id, g);
            }
        }
    }
    Gradients { map: grads }
}
