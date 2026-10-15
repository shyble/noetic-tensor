//! The single-device probe of the E2 differential gate (`single_device_diff.sh` copies it into
//! `examples/` of the reference tree and of the tree under test, builds it in each and compares
//! the outputs byte for byte).
//!
//! Six training configurations, 60 steps each, 3 models on the seed axis: a decoder, a
//! mixture-of-experts decoder and a decoder with dropout 0.1, each under AdamW and under SGD with
//! momentum; per configuration it prints the sha256 of the final vars' saved form and of the bits
//! of the logits on a fixed batch, and per step the per-seed losses' bits are folded into a sha256.
//!
//! `cargo run --release --example single_device_probe [cpu|metal|cuda]`

use noetic::nn::dist::state_hash;
use noetic::nn::{train_step, Adam, AdamConfig, AuxWeights, Decoder, DecoderConfig, ForwardOptions, Optimizer, ParamGroups, Sgd, SgdConfig, VarMap};
use noetic::tensor::{set_default_device, CpuMode, Device, IntTensor, Tensor};
use noetic::rng::derive;
use rand::Rng;
use sha2::{Digest, Sha256};

fn hex(b: &[u8]) -> String {
    Sha256::digest(b).iter().map(|x| format!("{x:02x}")).collect()
}

/// A batch `[S, B, T]` of random tokens (targets the next token, every third position masked).
fn batch(cfg: &DecoderConfig, s: usize, b: usize, seed: u64) -> (IntTensor, IntTensor, Tensor) {
    let t = cfg.context;
    let mut r = derive(seed, "probe batch");
    let (mut tk, mut tg, mut mk) = (vec![], vec![], vec![]);
    for _ in 0..s * b {
        let seq: Vec<i64> = (0..=t).map(|_| r.gen_range(0..cfg.vocab as i64)).collect();
        tk.extend_from_slice(&seq[..t]);
        tg.extend_from_slice(&seq[1..]);
        mk.extend((0..t).map(|i| if i % 3 == 0 { 0.0f32 } else { 1.0 }));
    }
    (IntTensor::from_data(tk, [s, b, t]), IntTensor::from_data(tg, [s, b, t]), Tensor::from_data(mk, [s, b, t]))
}

fn main() -> noetic::Result<()> {
    let device = match std::env::args().nth(1).as_deref() {
        Some("metal") => Device::Metal(0),
        Some("cuda") => Device::Cuda(0),
        _ => Device::Cpu(CpuMode::Reference),
    };
    let mut dropout = DecoderConfig::new(16, 32, 4, 8, 2, 64);
    dropout.dropout = 0.1;
    dropout.dropout_root = 3;
    let configs = [("decoder", DecoderConfig::new(16, 32, 4, 8, 2, 64)), ("moe", DecoderConfig::moe_small(20, 10)), ("dropout", dropout)];
    let (s, b, steps) = (3usize, 6usize, 60u64);
    for (name, cfg) in configs {
        for opt_name in ["adamw", "sgd"] {
            set_default_device(Device::Cpu(CpuMode::Reference))?;
            let (_, cpu) = Decoder::init(cfg.clone(), s, 7)?;
            let mut vars = VarMap::new();
            for (n, v) in cpu.iter() {
                vars.insert(n, v.clone().to(device))?;
            }
            let mut opt: Box<dyn Optimizer> = match opt_name {
                "adamw" => Box::new(Adam::new(AdamConfig::adamw(3e-3, 0.01, Default::default()), ParamGroups::new(&vars), &vars)),
                _ => Box::new(Sgd::new(SgdConfig { lr: 0.05, momentum: 0.9, weight_decay: 1e-3, ..Default::default() }, ParamGroups::new(&vars), &vars)),
            };
            set_default_device(device)?;
            let mut losses = vec![];
            for step in 0..steps {
                let (tk, tg, mk) = batch(&cfg, s, b, 100 + step);
                let r = train_step(&cfg, &mut vars, opt.as_mut(), &tk, &tg, &mk, step, AuxWeights::default(), 1.0);
                for x in r.ce {
                    losses.extend_from_slice(&x.to_bits().to_le_bytes());
                }
            }
            let (tk, _, _) = batch(&cfg, s, b, 9999);
            let model = Decoder::load(cfg.clone(), &vars)?;
            let logits = model.forward_with(&tk, &ForwardOptions { pad: None, train_step: None }).logits.to_vec();
            let lb: Vec<u8> = logits.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect();
            println!("{name} {opt_name}: vars {} logits {} losses {}", state_hash(&vars)?, hex(&lb), hex(&losses));
        }
    }
    Ok(())
}
