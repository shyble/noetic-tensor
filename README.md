# noetic-tensor

**A deterministic deep-learning engine in Rust: tensors, autodiff and transformer building blocks whose runs are bit-for-bit reproducible on the same machine.**

noetic-tensor is a small, self-contained framework written from scratch. It has its own tensor type, a reverse-mode autodiff and a neural-network library with the parts a modern decoder needs: grouped-query attention, RoPE, a KV cache, RMSNorm, SwiGLU and mixture-of-experts layers, AdamW and learning-rate schedules. The same code runs on the CPU, on Apple Metal and on NVIDIA CUDA.

## Why it exists

Most frameworks treat run-to-run variation as noise you live with. noetic-tensor treats it as a bug:

- **Reproducible to the bit.**
  - The reference CPU backend computes every operation in a fixed order: reductions, matmul (pinned to one gemm kernel), the backward pass.
  - The same inputs and seed give byte-identical weights, logits and gradients on every run.
  - A process can pin itself to that backend, so nothing selects a faster, non-identical path behind its back.
  - Results are reproducible per platform and device; across platforms and between CPU and GPU they agree within stated tolerances.
- **Many models in one pass.**
  - Every parameter carries a leading seed axis `[S, …]`, so S independently initialised models train side by side in one forward and backward pass.
  - Seed i's weights, logits and gradients never depend on the other seeds.
  - That makes multi-seed experiments and per-seed learning-rate grids cheap and exact.
- **Small enough to read.**
  - Six dependencies: `matrixmultiply`, `rand`, `rand_chacha`, `serde`, `serde_json`, `sha2`.
  - The Metal and CUDA backends call the system libraries through hand-written FFI, with no GPU crates.
  - The GPU kernels (the Metal shaders and the CUDA source) are in this repository; the CPU matmul is `matrixmultiply`'s, pinned to one version.
- **Checked, not assumed.**
  - Operations are tested against recorded reference outputs, f64 references and finite differences.
  - Every GPU backend is tested against the CPU reference.
  - The claims below are exactly what the tests check.

It suits research code whose results must be rebuilt exactly, experiments that run many seeds of a small model, and anyone who wants to see how a framework works from the tensor up.

This repository is the engine's single source: releases are tagged here, starting with 0.2.0, and their changes are in `CHANGELOG.md`. This crate is the tensor and neural-network engine only; it contains none of Noetic's methods. The library is imported as `use noetic::tensor::…` and `use noetic::nn::…` (the package name stays noetic-tensor).

## Using it

```toml
[dependencies]
noetic-tensor = { git = "https://github.com/shyble/noetic-tensor" }
# GPU backends are opt-in:
# noetic-tensor = { git = "https://github.com/shyble/noetic-tensor", features = ["metal"] }   # macOS
# noetic-tensor = { git = "https://github.com/shyble/noetic-tensor", features = ["cuda"] }    # CUDA toolkit at CUDA_PATH
```

A small decoder, four models at once:

```rust
use noetic::nn::{cross_entropy, Decoder, DecoderConfig};
use noetic::tensor::IntTensor;

// vocab 16, width 32, 4 heads, context 8, 2 blocks, MLP width 64; 4 seeds from root 1.
let cfg = DecoderConfig::new(16, 32, 4, 8, 2, 64);
let (model, _vars) = Decoder::init(cfg, 4, 1)?;
let tokens = IntTensor::from_data(vec![1; 4 * 2 * 8], [4, 2, 8]); // [seeds, batch, time]
let logits = model.forward(&tokens);                               // [4, 2, 8, 16]
let loss = cross_entropy(logits, &tokens);                         // one loss per seed, [4]
assert_eq!(loss.shape(), &[4]);
# Ok::<(), noetic::Error>(())
```

## What is in it

**`tensor`**
- Float, int and bool tensors: F32, F64, F16 and BF16 storage (16-bit types compute in f32), I64, I32 and U8.
- Views (reshape, swap_dims, slice, expand), broadcasting, batched matmul, reductions, indexing, gather (with a scatter-add backward), sort, argmax.
- A tape-based reverse-mode autodiff.
- A panicking API and a fallible `try_*` counterpart for every check.
- Devices:
  - `Cpu(Reference)`: deterministic reference kernels.
  - `Cpu(Fast)`: worker threads and SIMD.
  - `Metal(i)`: feature `metal`, macOS only.
  - `Cuda(i)`: feature `cuda`. Uses the driver API, NVRTC and cuBLAS.

**`nn`**, structured like candle-nn:
- Layers: linear, embedding (one-hot or gather), multi-head and grouped-query attention, RoPE, a KV cache, RMSNorm and LayerNorm, gated (SwiGLU) and plain MLPs, mixture of experts, dropout.
- A pre-norm decoder. Its gated MLP takes an optional caller-supplied extension (`MlpExtension`) that may add vars and transform the gate's pre-activation and the hidden activations.
- Losses.
- Optimizers: Adam and AdamW (both decay orders), SGD, EMA, learning-rate schedules, gradient clipping.
- `VarMap`: save and load, exact to the bit.
- Data-parallel training across processes (tested on one machine; multi-machine runs are untested) (`nn::dist`), in the shape of torch.distributed, DDP and torchrun: a process group from torchrun's environment variables, collectives over TCP (standard library only), a deterministic mode in which gradients are summed in rank order (W processes with k accumulation micro-steps each give the weights of one process with W·k, bit for bit, on the CPU, tested on one machine), a sharded sampler, coordinated checkpoints and a launcher (`examples/launch.rs`). See `doc/distributed.md`.

Every var has a leading seed axis `[S, …]`. That lets S independently initialised models train in one pass, and seed i's weights never depend on S.

## Example

```sh
cargo run --release --example copy_task            # CPU reference backend
cargo run --release --features metal --example copy_task metal
```

```rust
use noetic::tensor::Tensor;

let x = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], [2, 2]).require_grad();
let y = (x.clone().matmul(x.clone()) * 2.0).sum();
let grads = y.backward();
println!("{:?}", x.grad(&grads).unwrap().to_vec());
```

## Determinism and accuracy, as the tests check them

- **`Cpu(Reference)`**
  - Byte-identical run to run.
  - For the operations whose numerics were kept, it is bit-equal to burn 0.21's recorded outputs on macOS aarch64 and Windows x86_64 (`src/tensor/tests/fixtures`).
  - The other operations use the standard formulations. Their gradients are checked against f64 and central finite differences.
- **`Cpu(Fast)`**
  - Bit-equal to Reference for elementwise maps, matmul and reductions across lanes.
  - Within 1e-5 relative for lane reductions, and for a whole decoder's loss.
  - Within 1e-4 for its gradients.
- **Metal and CUDA**
  - Decoder logits and loss within 1e-5 of Reference, and gradients within 1e-4. Relative error is |Δ|/(1+|ref|).
  - Greedy tokens identical except after a step whose top two logits tie.
  - A 200-step training trajectory stays inside the envelope of a 16-member ±1/±2-ulp CPU ensemble.
  - Byte-identical run to run on the same device.
- **Seed axis:** seed i's initial weights, logits and gradients are bit-equal whatever the other seeds are.

## Tests

```sh
cargo test --release                      # CPU
cargo test --release --features metal     # plus the Metal bodies (macOS)
cargo test --release --features cuda      # plus the CUDA bodies
```

The GPU test bodies run in a fresh process, because the reference pin (`tensor::pin_reference`) is process-wide. Benchmarks and profiles are `#[ignore]`d: run them with `-- --ignored --nocapture`.

## Acknowledgements

The reference kernels were first written to reproduce [burn](https://github.com/tracel-ai/burn) 0.21 (`Autodiff<NdArray<f32>>`) bit for bit, and the autodiff traversal follows burn-autodiff's. The fixtures in `src/tensor/tests/fixtures` are outputs recorded from burn 0.21. burn is dual-licensed under MIT and Apache-2.0.

## Licence

MIT or Apache-2.0, at your option (LICENSE-MIT, LICENSE-APACHE).
