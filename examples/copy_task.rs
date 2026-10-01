//! Train a small decoder to copy three symbols across a separator (`a b c SEP a b c`), four
//! independently initialised models at once on the seed axis, and print each one's answer loss.
//!
//! `cargo run --release --example copy_task` (add `--features metal` or `--features cuda` and
//! pass `metal` or `cuda` to train on the GPU).

use noetic_tensor::nn::{train_step, Adam, AdamConfig, AuxWeights, Decoder, DecoderConfig, ParamGroups, VarMap};
use noetic_tensor::tensor::{set_default_device, CpuMode, Device, IntTensor, Tensor};
use rand::{Rng, SeedableRng};

fn main() -> noetic_tensor::Result<()> {
    let device = match std::env::args().nth(1).as_deref() {
        Some("metal") => Device::Metal(0),
        Some("cuda") => Device::Cuda(0),
        Some("fast") => Device::Cpu(CpuMode::Fast),
        _ => Device::Cpu(CpuMode::Reference),
    };
    set_default_device(device)?;

    // Vocabulary of 16: 0 is padding, 1 the separator, 2..16 the symbols.
    let cfg = DecoderConfig::new(16, 32, 4, 7, 2, 96);
    let (seeds, batch, t) = (4, 32, 7);
    // Vars are created on the CPU; move them to the device.
    let (_, cpu_vars) = Decoder::init(cfg.clone(), seeds, 1)?;
    let mut vars = VarMap::new();
    for (name, v) in cpu_vars.iter() {
        vars.insert(name, v.clone().to(device))?;
    }
    let mut opt = Adam::new(AdamConfig::default(), ParamGroups::new(&vars), &vars);
    println!("{} parameters per model, {seeds} models, on {device:?}", vars.count_per_seed());

    let mut rng = rand::rngs::StdRng::seed_from_u64(7);
    for step in 0..400u64 {
        let (mut toks, mut targets) = (Vec::new(), Vec::new());
        for _ in 0..seeds * batch {
            let a: Vec<i64> = (0..3).map(|_| rng.gen_range(2..16)).collect();
            let seq = [a.clone(), vec![1], a, vec![0]].concat();
            toks.extend_from_slice(&seq[..t]);
            targets.extend_from_slice(&seq[1..]);
        }
        // Score only the answer: positions 3..6 predict the copied symbols.
        let mask: Vec<f32> = (0..seeds * batch * t).map(|i| if (3..6).contains(&(i % t)) { 1.0 } else { 0.0 }).collect();
        let report = train_step(
            &cfg,
            &mut vars,
            &mut opt,
            &IntTensor::from_data(toks, [seeds, batch, t]),
            &IntTensor::from_data(targets, [seeds, batch, t]),
            &Tensor::from_data(mask, [seeds, batch, t]),
            step,
            AuxWeights::default(),
            1.0,
        );
        if step % 50 == 0 || step == 399 {
            let losses: Vec<String> = report.ce.iter().map(|l| format!("{l:.3}")).collect();
            println!("step {step:>3}: answer loss per model [{}]", losses.join(", "));
        }
    }
    Ok(())
}
