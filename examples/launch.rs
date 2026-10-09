//! A torchrun-style launcher: start N processes of a command on this node with the rendezvous
//! variables set (MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE, LOCAL_RANK, LOCAL_WORLD_SIZE,
//! GROUP_RANK), and stop the whole job when one of them fails.
//!
//! ```sh
//! cargo build --release --examples
//! target/release/examples/launch --nproc-per-node 2 -- target/release/examples/ddp_copy_task
//! # two nodes: the same command on each, with --nnodes 2 --node-rank <i> --master-addr <node 0>
//! ```
//!
//! Options: `--nproc-per-node N` (1), `--nnodes M` (1), `--node-rank I` (0), `--master-addr A`
//! (127.0.0.1), `--master-port P` (a free port; set it on every node of a multi-node job),
//! `--log-dir D` (one log per rank; default: this terminal).

use noetic::nn::dist::{free_port, run, LaunchConfig};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let split = args.iter().position(|a| a == "--").unwrap_or_else(|| usage("no `--` before the command"));
    let (opts, cmd) = (&args[..split], &args[split + 1..]);
    if cmd.is_empty() {
        usage("no command after `--`");
    }
    let mut cfg = LaunchConfig {
        nproc_per_node: 1,
        nnodes: 1,
        node_rank: 0,
        master_addr: "127.0.0.1".into(),
        master_port: 0,
        command: cmd[0].clone().into(),
        args: cmd[1..].to_vec(),
        env: vec![],
        log_dir: None,
    };
    let mut it = opts.iter();
    while let Some(k) = it.next() {
        let v = it.next().unwrap_or_else(|| usage(&format!("{k} needs a value")));
        let num = |v: &str| v.parse::<usize>().unwrap_or_else(|_| usage(&format!("{k} {v}: not a number")));
        match k.as_str() {
            "--nproc-per-node" => cfg.nproc_per_node = num(v),
            "--nnodes" => cfg.nnodes = num(v),
            "--node-rank" => cfg.node_rank = num(v),
            "--master-addr" => cfg.master_addr = v.clone(),
            "--master-port" => cfg.master_port = u16::try_from(num(v)).unwrap_or_else(|_| usage("--master-port: not a port")),
            "--log-dir" => cfg.log_dir = Some(v.into()),
            _ => usage(&format!("unknown option {k}")),
        }
    }
    if cfg.master_port == 0 {
        if cfg.nnodes > 1 {
            usage("a multi-node job needs --master-port");
        }
        cfg.master_port = free_port().unwrap_or_else(|e| fail(&e.to_string()));
    }
    eprintln!("launch: {} process(es) on node {} of {}, world size {}, rendezvous {}:{}", cfg.nproc_per_node, cfg.node_rank, cfg.nnodes, cfg.world_size(), cfg.master_addr, cfg.master_port);
    if let Err(e) = run(&cfg) {
        fail(&e.to_string());
    }
}

fn usage(why: &str) -> ! {
    eprintln!("launch: {why}\nusage: launch [--nproc-per-node N] [--nnodes M --node-rank I --master-addr A --master-port P] [--log-dir D] -- <command> [args...]");
    std::process::exit(2)
}

fn fail(why: &str) -> ! {
    eprintln!("launch: {why}");
    std::process::exit(1)
}
