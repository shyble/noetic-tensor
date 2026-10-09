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
//! `--log-dir D` (one log per rank; default: this terminal), `--secret-file F` (the job's secret
//! in hex, at least 32 bytes; the same file on every node), `--no-auth` (no secret: isolated
//! development networks only). A single-node job without a secret file gets a fresh secret.

use noetic::nn::dist::{free_port, run, JobSecret, LaunchConfig};

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
        secret: None,
    };
    let (mut secret_file, mut no_auth) = (None, false);
    let mut it = opts.iter();
    while let Some(k) = it.next() {
        if k == "--no-auth" {
            no_auth = true;
            continue;
        }
        let v = it.next().unwrap_or_else(|| usage(&format!("{k} needs a value")));
        let num = |v: &str| v.parse::<usize>().unwrap_or_else(|_| usage(&format!("{k} {v}: not a number")));
        match k.as_str() {
            "--nproc-per-node" => cfg.nproc_per_node = num(v),
            "--nnodes" => cfg.nnodes = num(v),
            "--node-rank" => cfg.node_rank = num(v),
            "--master-addr" => cfg.master_addr = v.clone(),
            "--master-port" => cfg.master_port = u16::try_from(num(v)).unwrap_or_else(|_| usage("--master-port: not a port")),
            "--log-dir" => cfg.log_dir = Some(v.into()),
            "--secret-file" => secret_file = Some(v.clone()),
            _ => usage(&format!("unknown option {k}")),
        }
    }
    if cfg.master_port == 0 {
        if cfg.nnodes > 1 {
            usage("a multi-node job needs --master-port");
        }
        cfg.master_port = free_port().unwrap_or_else(|e| fail(&e.to_string()));
    }
    cfg.secret = match job_secret(secret_file.as_deref(), no_auth, cfg.nnodes) {
        Ok(s) => s,
        Err(Refused::Usage(why)) => usage(&why),
        Err(Refused::Fail(why)) => fail(&why),
    };
    if cfg.secret.is_none() {
        eprintln!("launch: no job secret: unauthenticated, for an isolated development network only");
    }
    eprintln!("launch: {} process(es) on node {} of {}, world size {}, rendezvous {}:{}", cfg.nproc_per_node, cfg.node_rank, cfg.nnodes, cfg.world_size(), cfg.master_addr, cfg.master_port);
    if let Err(e) = run(&cfg) {
        fail(&e.to_string());
    }
}

/// Why the launch is refused: a usage error (exit 2) or a failure (exit 1).
#[derive(Debug)]
enum Refused {
    Usage(String),
    Fail(String),
}

/// The job's secret: from `--secret-file`, none with `--no-auth`, a fresh one for a single-node
/// job; a multi-node job without either is refused.
fn job_secret(secret_file: Option<&str>, no_auth: bool, nnodes: usize) -> Result<Option<JobSecret>, Refused> {
    match (secret_file, no_auth) {
        (Some(_), true) => Err(Refused::Usage("--secret-file and --no-auth exclude each other".into())),
        (Some(f), false) => {
            let text = std::fs::read_to_string(f).map_err(|e| Refused::Fail(format!("{f}: {e}")))?;
            JobSecret::from_hex(&text).map(Some).map_err(|e| Refused::Fail(format!("{f}: {e}")))
        }
        (None, true) => Ok(None),
        (None, false) if nnodes == 1 => Ok(Some(JobSecret::random())),
        (None, false) => Err(Refused::Usage("a multi-node job needs --secret-file (or --no-auth on an isolated development network)".into())),
    }
}

fn usage(why: &str) -> ! {
    eprintln!("launch: {why}\nusage: launch [--nproc-per-node N] [--nnodes M --node-rank I --master-addr A --master-port P] [--log-dir D] [--secret-file F | --no-auth] -- <command> [args...]");
    std::process::exit(2)
}

fn fail(why: &str) -> ! {
    eprintln!("launch: {why}");
    std::process::exit(1)
}

#[cfg(test)]
mod tests {
    use super::{job_secret, Refused};

    #[test]
    fn a_multi_node_launch_without_a_secret_is_refused() {
        for nnodes in [2, 3] {
            let e = job_secret(None, false, nnodes).unwrap_err();
            assert!(matches!(&e, Refused::Usage(m) if m.contains("multi-node job needs --secret-file")), "{e:?}");
        }
        // One node gets a fresh secret; --no-auth is an explicit choice, on any number of nodes.
        assert!(job_secret(None, false, 1).unwrap().is_some());
        assert!(job_secret(None, true, 2).unwrap().is_none());
        assert!(matches!(job_secret(Some("f"), true, 2), Err(Refused::Usage(_))));
        // A secret file shared by the nodes is read; a missing or short one is refused.
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("target").join("launch-tests");
        std::fs::create_dir_all(&dir).unwrap();
        let (good, short) = (dir.join(format!("secret-{}", std::process::id())), dir.join(format!("short-{}", std::process::id())));
        std::fs::write(&good, "ab".repeat(32)).unwrap();
        std::fs::write(&short, "ab".repeat(8)).unwrap();
        let got = job_secret(Some(good.to_str().unwrap()), false, 2).unwrap().unwrap();
        assert_eq!(got.to_hex(), "ab".repeat(32));
        assert!(matches!(job_secret(Some(short.to_str().unwrap()), false, 2), Err(Refused::Fail(_))));
        assert!(matches!(job_secret(Some(dir.join("missing").to_str().unwrap()), false, 2), Err(Refused::Fail(_))));
        for p in [good, short] {
            std::fs::remove_file(p).unwrap();
        }
    }
}
