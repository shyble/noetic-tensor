//! The process group: the rendezvous, a full mesh of TCP connections, and the collectives in
//! deterministic mode.
//!
//! **Rendezvous.** Rank 0 listens on `MASTER_ADDR:MASTER_PORT`. Every other rank connects, says
//! its rank, the world size, the job key and the address of its own data listener (on the
//! interface it reached rank 0 through). Rank 0 checks that the ranks are distinct and agree on
//! the world size and the job key, then sends every rank the table of data addresses. Each rank
//! connects to every lower rank and accepts every higher one, and a barrier closes the
//! rendezvous. A mismatch refuses the whole job with the reason on every rank.
//!
//! **Collectives.** Every collective has a sequence number; each message names its operation and
//! number, so ranks that call different collectives fail instead of mixing data.
//! - `barrier`, `broadcast` and `all_gather` go through rank 0 (or the root), in rank order.
//! - `all_reduce_grads` (and `all_reduce_f32`) pass the running sum along the ranks in order:
//!   rank 0 sends its sum to rank 1, which folds its own micro-steps in and sends it on, and the
//!   last rank sends the total to every other rank. The sum is therefore the fold over the global
//!   micro-steps in rank order, whatever the timing. Large sums go in chunks so the ranks work
//!   at once; the chunk size changes the pipelining, never the order of the additions.

use super::auth::{self, JobSecret};
use super::env::DistEnv;
use super::reduce::{fold, fold_var, GradSum, Part};
use super::wire::{self, Dec, Enc, Op};
use crate::error::{NnError, Result};
use crate::nn::VarMap;
use std::net::{SocketAddr, TcpListener, TcpStream, ToSocketAddrs};
use std::time::{Duration, Instant};

/// Options of a process group.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GroupOptions {
    /// A string every rank must agree on (e.g. a hash of the run's configuration); a rank with
    /// another key is refused at the rendezvous.
    pub job_key: String,
    /// How long a collective waits for a peer's message before failing.
    pub timeout: Duration,
    /// How long the rendezvous waits for every rank.
    pub connect_timeout: Duration,
    /// The all-reduce's chunk, in elements (pipelining only; the sum's order does not depend on it).
    pub chunk_elems: usize,
    /// The platform key every rank must share (`platform_key()`; leave the default).
    pub platform: String,
    /// The job's secret for the authenticated handshake; None reads `DIST_JOB_SECRET` in
    /// `from_env`, and without one the job is unauthenticated (isolated development networks
    /// only).
    pub secret: Option<JobSecret>,
}

impl Default for GroupOptions {
    fn default() -> Self {
        GroupOptions { job_key: String::new(), timeout: Duration::from_secs(600), connect_timeout: Duration::from_secs(120), chunk_elems: 1 << 20, platform: super::env::platform_key(), secret: None }
    }
}

/// A process's group: its rank, the world size and its connections to every other rank.
#[derive(Debug)]
pub struct ProcessGroup {
    env: DistEnv,
    opts: GroupOptions,
    /// By rank; None for this process.
    peers: Vec<Option<TcpStream>>,
    seq: u64,
}

fn dist<E: std::fmt::Display>(what: impl std::fmt::Display) -> impl FnOnce(E) -> NnError {
    move |e| NnError::Dist(format!("{what}: {e}"))
}

fn resolve(host: &str, port: u16) -> Result<SocketAddr> {
    let addrs: Vec<SocketAddr> = (host, port).to_socket_addrs().map_err(dist(format!("resolving {host}:{port}")))?.collect();
    addrs.iter().find(|a| a.is_ipv4()).or(addrs.first()).copied().ok_or_else(|| NnError::Dist(format!("{host}:{port} resolves to no address")))
}

fn connect(addr: SocketAddr, deadline: Instant, what: &str) -> Result<TcpStream> {
    loop {
        let left = deadline.saturating_duration_since(Instant::now());
        match TcpStream::connect_timeout(&addr, left.clamp(Duration::from_millis(1), Duration::from_secs(1))) {
            Ok(s) => return Ok(s),
            Err(e) if left.is_zero() => return Err(NnError::Dist(format!("connecting to {what} at {addr}: {e} (timed out)"))),
            Err(_) => std::thread::sleep(Duration::from_millis(20)),
        }
    }
}

fn accept(l: &TcpListener, deadline: Instant, what: &str) -> Result<TcpStream> {
    l.set_nonblocking(true).map_err(dist("listener"))?;
    loop {
        match l.accept() {
            Ok((s, _)) => {
                // Accepted sockets inherit the listener's non-blocking flag on some systems.
                s.set_nonblocking(false).map_err(dist("socket"))?;
                return Ok(s);
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                if Instant::now() >= deadline {
                    return Err(NnError::Dist(format!("waiting for {what}: timed out")));
                }
                std::thread::sleep(Duration::from_millis(5));
            }
            Err(e) => return Err(NnError::Dist(format!("waiting for {what}: {e}"))),
        }
    }
}

fn tune(s: &TcpStream, timeout: Duration) -> Result<()> {
    s.set_nodelay(true).map_err(dist("socket"))?;
    s.set_read_timeout(Some(timeout.max(Duration::from_millis(1)))).map_err(dist("socket"))?;
    s.set_write_timeout(Some(timeout.max(Duration::from_millis(1)))).map_err(dist("socket"))?;
    Ok(())
}

/// One chunk of the all-reduce: (var, first element, end) segments.
type Chunk = Vec<(usize, usize, usize)>;

/// Cut the vars' elements into chunks of at most `max` elements (a var may span chunks; an
/// empty var is a segment of its own, so its presence still travels).
fn plan(sizes: &[usize], max: usize) -> Vec<Chunk> {
    let max = max.max(1);
    let mut out = vec![];
    let mut cur: Chunk = vec![];
    let mut room = max;
    for (v, &n) in sizes.iter().enumerate() {
        if n == 0 {
            cur.push((v, 0, 0));
            continue;
        }
        let mut lo = 0;
        while lo < n {
            let take = room.min(n - lo);
            cur.push((v, lo, lo + take));
            lo += take;
            room -= take;
            if room == 0 {
                out.push(std::mem::take(&mut cur));
                room = max;
            }
        }
    }
    if !cur.is_empty() || out.is_empty() {
        out.push(cur);
    }
    out
}

/// Chunk `c` of a whole gradient set.
fn encode_chunk(c: usize, segs: &Chunk, part: &Part) -> Vec<u8> {
    let vals: Vec<Option<&[f32]>> = segs.iter().map(|&(v, lo, hi)| part[v].as_ref().map(|x| &x[lo..hi])).collect();
    encode(c, &vals)
}

/// Chunk `c` from its segments' values.
fn encode_vals(c: usize, vals: &[Option<Vec<f32>>]) -> Vec<u8> {
    encode(c, &vals.iter().map(|x| x.as_deref()).collect::<Vec<_>>())
}

fn encode(c: usize, vals: &[Option<&[f32]>]) -> Vec<u8> {
    let mut e = Enc::default();
    e.u64(c as u64);
    for x in vals {
        match x {
            Some(x) => {
                e.u8(1).f32s(x);
            }
            None => {
                e.u8(0);
            }
        }
    }
    e.0
}

fn decode_chunk(c: usize, segs: &Chunk, payload: &[u8]) -> Result<Vec<Option<Vec<f32>>>> {
    let mut d = Dec::new(payload);
    let got = d.u64()? as usize;
    if got != c {
        return Err(NnError::Dist(format!("all-reduce chunk {got} where chunk {c} was due")));
    }
    let mut out = Vec::with_capacity(segs.len());
    for &(_, lo, hi) in segs {
        out.push(match d.u8()? {
            0 => None,
            1 => Some(d.f32s(hi - lo)?),
            b => return Err(NnError::Dist(format!("all-reduce chunk {c}: presence byte {b}"))),
        });
    }
    d.done()?;
    Ok(out)
}

impl ProcessGroup {
    /// The group of one process: no network; every collective is the identity.
    pub fn single() -> ProcessGroup {
        ProcessGroup { env: DistEnv::single(), opts: GroupOptions::default(), peers: vec![None], seq: 0 }
    }

    /// The group the environment describes (`DistEnv::from_env`), or the single-process group
    /// when distribution is off.
    pub fn from_env(mut opts: GroupOptions) -> Result<ProcessGroup> {
        if opts.secret.is_none() {
            opts.secret = JobSecret::from_env()?;
        }
        match DistEnv::from_env()? {
            None => Ok(ProcessGroup { opts, ..ProcessGroup::single() }),
            Some(env) => ProcessGroup::init(&env, opts),
        }
    }

    /// Join the group `env` describes: the rendezvous, the mesh and a barrier.
    pub fn init(env: &DistEnv, opts: GroupOptions) -> Result<ProcessGroup> {
        let (rank, world) = (env.rank, env.world_size);
        if world == 0 || rank >= world {
            return Err(NnError::Dist(format!("rank {rank} is not in 0..{world}")));
        }
        if world == 1 {
            return Ok(ProcessGroup { env: env.clone(), opts, peers: vec![None], seq: 0 });
        }
        let deadline = Instant::now() + opts.connect_timeout;
        let master = resolve(&env.master_addr, env.master_port)?;
        let me = |what: &str| format!("rank {rank}: {what}");
        let (data, table) = if rank == 0 {
            let l = TcpListener::bind(master).map_err(dist(me(&format!("listening on {master}"))))?;
            let data = TcpListener::bind((master.ip(), 0)).map_err(dist(me("data listener")))?;
            let mut table = vec![String::new(); world];
            table[0] = TcpListener::local_addr(&data).map_err(dist("listener"))?.to_string();
            let mut joined: Vec<Option<TcpStream>> = (0..world).map(|_| None).collect();
            let mut refusal = None;
            let secret = opts.secret.as_ref();
            // Each rank's challenge, answered in the table.
            let mut answers: Vec<Vec<u8>> = vec![vec![]; world];
            let mut extra: Vec<(TcpStream, Vec<u8>)> = vec![];
            for _ in 1..world {
                let mut s = accept(&l, deadline, "the ranks to join")?;
                tune(&s, deadline.saturating_duration_since(Instant::now()))?;
                let mine = auth::challenge();
                let mut e = Enc::default();
                e.bytes(&mine);
                wire::send(&mut s, Op::Challenge, 0, 0, &e.0)?;
                let (op, _, _, payload) = wire::recv_any(&mut s, 0, usize::MAX)?;
                let mut d = Dec::new(&payload);
                let (r, w, key, platform, addr, theirs) = (d.u64()? as usize, d.u64()? as usize, d.str()?, d.str()?, d.str()?, d.bytes()?.to_vec());
                let fields = payload[..d.at()].to_vec();
                let proof = d.bytes()?.to_vec();
                d.done()?;
                let why = if op != Op::Hello {
                    Some("a connection that is not a rank".to_string())
                } else if !auth::verify(secret, "hello", &mine, &fields, &proof) {
                    Some(format!("a rank calling itself rank {r} failed the authenticated handshake (another job secret, or none where one is needed)"))
                } else if w != world {
                    Some(format!("rank {r} has world size {w}, rank 0 has {world}"))
                } else if key != opts.job_key {
                    Some(format!("rank {r} has job key {key:?}, rank 0 has {:?}", opts.job_key))
                } else if platform != opts.platform {
                    Some(format!("rank {r} runs on platform {platform:?}, rank 0 on {:?} (a mixed job)", opts.platform))
                } else if r == 0 || r >= world || joined[r].is_some() {
                    Some(format!("a second rank {r} (or one outside 1..{world})"))
                } else {
                    None
                };
                if let Some(why) = why {
                    refusal.get_or_insert(why);
                    // Keep the connection so the refused rank hears the reason.
                    extra.push((s, theirs));
                    continue;
                }
                table[r] = addr;
                joined[r] = Some(s);
                answers[r] = theirs;
            }
            let mut e = Enc::default();
            match &refusal {
                Some(why) => {
                    e.u8(1).str(why);
                }
                None => {
                    e.u8(0);
                    for a in &table {
                        e.str(a);
                    }
                }
            }
            // The table (or the refusal) as one field, and the answer to the rank's challenge.
            let table_msg = |theirs: &[u8]| {
                let mut m = Enc::default();
                m.bytes(&e.0).bytes(&auth::answer(secret, "table", theirs, &e.0));
                m.0
            };
            for (r, s) in joined.iter_mut().enumerate() {
                if let Some(s) = s {
                    let _ = wire::send(s, Op::Table, 0, 0, &table_msg(&answers[r]));
                }
            }
            for (s, theirs) in extra.iter_mut() {
                let _ = wire::send(s, Op::Table, 0, 0, &table_msg(theirs));
            }
            if let Some(why) = refusal {
                return Err(NnError::Dist(format!("the rendezvous refused the job: {why}")));
            }
            (data, table)
        } else {
            let mut s = connect(master, deadline, "rank 0")?;
            tune(&s, deadline.saturating_duration_since(Instant::now()).max(Duration::from_secs(1)))?;
            let ip = TcpStream::local_addr(&s).map_err(dist("socket"))?.ip();
            let data = TcpListener::bind((ip, 0)).map_err(dist(me("data listener")))?;
            let secret = opts.secret.as_ref();
            let payload = wire::recv(&mut s, Op::Challenge, rank, 0, 0)?;
            let mut d = Dec::new(&payload);
            let theirs = d.bytes()?.to_vec();
            d.done()?;
            let mine = auth::challenge();
            let mut e = Enc::default();
            e.u64(rank as u64).u64(world as u64).str(&opts.job_key).str(&opts.platform).str(&TcpListener::local_addr(&data).map_err(dist("listener"))?.to_string()).bytes(&mine);
            let proof = auth::answer(secret, "hello", &theirs, &e.0);
            e.bytes(&proof);
            wire::send(&mut s, Op::Hello, rank, 0, &e.0)?;
            let payload = wire::recv(&mut s, Op::Table, rank, 0, 0)?;
            let mut outer = Dec::new(&payload);
            let (fields, proof) = (outer.bytes()?, outer.bytes()?);
            outer.done()?;
            let authentic = auth::verify(secret, "table", &mine, fields, proof);
            let mut d = Dec::new(fields);
            if d.u8()? != 0 {
                let why = d.str()?;
                let tag = if authentic { "" } else { " (an unauthenticated reply)" };
                return Err(NnError::Dist(format!("the rendezvous refused the job{tag}: {why}")));
            }
            if !authentic {
                return Err(NnError::Dist(format!("rank {rank}: rank 0's reply failed the authenticated handshake (another job secret, or none where one is needed)")));
            }
            let table: Vec<String> = (0..world).map(|_| d.str()).collect::<Result<_>>()?;
            d.done()?;
            (data, table)
        };
        // The mesh: connect to every lower rank, accept every higher one.
        let mut peers: Vec<Option<TcpStream>> = (0..world).map(|_| None).collect();
        let secret = opts.secret.as_ref();
        for (q, addr) in table.iter().enumerate().take(rank) {
            let a: SocketAddr = addr.parse().map_err(dist(format!("rank {q}'s address {addr:?}")))?;
            let mut s = connect(a, deadline, &format!("rank {q}"))?;
            tune(&s, deadline.saturating_duration_since(Instant::now()).max(Duration::from_secs(1)))?;
            let payload = wire::recv(&mut s, Op::Challenge, rank, q, 0)?;
            let mut d = Dec::new(&payload);
            let theirs = d.bytes()?.to_vec();
            d.done()?;
            let mine = auth::challenge();
            let mut e = Enc::default();
            e.u64(rank as u64).str(&opts.job_key).bytes(&mine);
            let proof = auth::answer(secret, "link", &theirs, &e.0);
            e.bytes(&proof);
            wire::send(&mut s, Op::Link, rank, 0, &e.0)?;
            let payload = wire::recv(&mut s, Op::LinkAck, rank, q, 0)?;
            let mut d = Dec::new(&payload);
            let back = d.bytes()?.to_vec();
            d.done()?;
            if !auth::verify(secret, "link answer", &mine, &(q as u64).to_le_bytes(), &back) {
                return Err(NnError::Dist(format!("rank {rank}: rank {q}'s link failed the authenticated handshake")));
            }
            peers[q] = Some(s);
        }
        for _ in rank + 1..world {
            let mut s = accept(&data, deadline, "the higher ranks to connect")?;
            tune(&s, deadline.saturating_duration_since(Instant::now()).max(Duration::from_secs(1)))?;
            let mine = auth::challenge();
            let mut e = Enc::default();
            e.bytes(&mine);
            wire::send(&mut s, Op::Challenge, rank, 0, &e.0)?;
            let (op, sender, _, payload) = wire::recv_any(&mut s, rank, usize::MAX)?;
            let mut d = Dec::new(&payload);
            let (q, key, theirs) = (d.u64()? as usize, d.str()?, d.bytes()?.to_vec());
            let fields = payload[..d.at()].to_vec();
            let proof = d.bytes()?.to_vec();
            d.done()?;
            if op != Op::Link || !auth::verify(secret, "link", &mine, &fields, &proof) {
                return Err(NnError::Dist(format!("rank {rank}: a link that failed the authenticated handshake")));
            }
            if q != sender || q <= rank || q >= world || peers[q].is_some() || key != opts.job_key {
                return Err(NnError::Dist(format!("rank {rank}: an unexpected link from rank {q}")));
            }
            let mut e = Enc::default();
            e.bytes(&auth::answer(secret, "link answer", &theirs, &(rank as u64).to_le_bytes()));
            wire::send(&mut s, Op::LinkAck, rank, 0, &e.0)?;
            peers[q] = Some(s);
        }
        for s in peers.iter().flatten() {
            tune(s, opts.timeout)?;
        }
        let mut g = ProcessGroup { env: env.clone(), opts, peers, seq: 0 };
        g.barrier()?;
        Ok(g)
    }

    pub fn rank(&self) -> usize {
        self.env.rank
    }

    pub fn world_size(&self) -> usize {
        self.env.world_size
    }

    pub fn env(&self) -> &DistEnv {
        &self.env
    }

    pub fn options(&self) -> &GroupOptions {
        &self.opts
    }

    /// Whether there is more than one process.
    pub fn is_distributed(&self) -> bool {
        self.world_size() > 1
    }

    /// The reproducibility key: mode, sum order, backend and the platform key every rank shares.
    /// The world size is not part of it: for a fixed number of global micro-steps the flat rank
    /// order gives the same bits at every world size, so runs at different world sizes under one
    /// key may be pooled once shown identical on that platform (see `record`).
    pub fn key(&self) -> String {
        format!("mode=deterministic order=flat-rank backend=tcp platform={}", self.opts.platform)
    }

    /// What a run records about its distribution: the key, the world size and whether the
    /// handshake was authenticated.
    pub fn record(&self) -> String {
        format!("{} world={} auth={}", self.key(), self.world_size(), if self.opts.secret.is_some() { "hmac-sha256" } else { "none" })
    }

    /// An empty gradient sum of the right kind for this rank (see `GradSum`).
    pub fn grad_sum(&self, vars: &VarMap) -> GradSum {
        if self.rank() == 0 {
            GradSum::new(vars)
        } else {
            GradSum::deferred(vars)
        }
    }

    fn peer(&mut self, q: usize) -> &mut TcpStream {
        self.peers[q].as_mut().expect("a peer, not this rank")
    }

    fn next_seq(&mut self) -> u64 {
        self.seq += 1;
        self.seq
    }

    /// Wait until every rank reaches the barrier.
    pub fn barrier(&mut self) -> Result<()> {
        let (r, w) = (self.rank(), self.world_size());
        if w == 1 {
            return Ok(());
        }
        let seq = self.next_seq();
        if r == 0 {
            for q in 1..w {
                wire::recv(self.peer(q), Op::Barrier, 0, q, seq)?;
            }
            for q in 1..w {
                wire::send(self.peer(q), Op::Barrier, 0, seq, &[])?;
            }
        } else {
            wire::send(self.peer(0), Op::Barrier, r, seq, &[])?;
            wire::recv(self.peer(0), Op::Barrier, r, 0, seq)?;
        }
        Ok(())
    }

    /// Replace `buf` on every rank with root's.
    pub fn broadcast(&mut self, root: usize, buf: &mut Vec<u8>) -> Result<()> {
        let (r, w) = (self.rank(), self.world_size());
        if root >= w {
            return Err(NnError::Dist(format!("broadcast from rank {root} in a world of {w}")));
        }
        if w == 1 {
            return Ok(());
        }
        let seq = self.next_seq();
        if r == root {
            for q in (0..w).filter(|q| *q != root) {
                let b = std::mem::take(buf);
                let res = wire::send(self.peer(q), Op::Broadcast, r, seq, &b);
                *buf = b;
                res?;
            }
        } else {
            *buf = wire::recv(self.peer(root), Op::Broadcast, r, root, seq)?;
        }
        Ok(())
    }

    /// Every rank's bytes, in rank order, on every rank.
    pub fn all_gather(&mut self, mine: &[u8]) -> Result<Vec<Vec<u8>>> {
        let (r, w) = (self.rank(), self.world_size());
        if w == 1 {
            return Ok(vec![mine.to_vec()]);
        }
        let seq = self.next_seq();
        let all = if r == 0 {
            let mut all = vec![mine.to_vec()];
            for q in 1..w {
                all.push(wire::recv(self.peer(q), Op::Gather, 0, q, seq)?);
            }
            let mut e = Enc::default();
            for a in &all {
                e.bytes(a);
            }
            for q in 1..w {
                wire::send(self.peer(q), Op::Gather, 0, seq, &e.0)?;
            }
            all
        } else {
            wire::send(self.peer(0), Op::Gather, r, seq, mine)?;
            let payload = wire::recv(self.peer(0), Op::Gather, r, 0, seq)?;
            let mut d = Dec::new(&payload);
            let all: Vec<Vec<u8>> = (0..w).map(|_| d.bytes().map(<[u8]>::to_vec)).collect::<Result<_>>()?;
            d.done()?;
            all
        };
        Ok(all)
    }

    /// The elementwise sum of every rank's `x`, added in rank order (`((x₀ + x₁) + x₂) + …`),
    /// on every rank.
    pub fn all_reduce_f32(&mut self, x: &[f32]) -> Result<Vec<f32>> {
        let layout = format!("an f32 vector of {}", x.len());
        let (mut acc, _) = self.chain(&[x.len()], &layout, vec![vec![Some(x.to_vec())]], 1)?;
        Ok(acc.pop().flatten().unwrap_or_default())
    }

    /// Sum every rank's micro-step gradients in global order (rank, then micro-step) and leave
    /// the total in `sum` on every rank, with the count of every rank's micro-steps.
    pub fn all_reduce_grads(&mut self, sum: &mut GradSum) -> Result<()> {
        if self.rank() > 0 && !sum.is_deferred() && sum.count() > 1 {
            return Err(NnError::Dist(format!(
                "rank {}: {} micro-steps were pre-added; a rank after the first must keep them apart (ProcessGroup::grad_sum)",
                self.rank(),
                sum.count()
            )));
        }
        let sizes: Vec<usize> = sum.slots().iter().map(|s| s.numel()).collect();
        let layout = sum.layout().to_string();
        let (parts, count) = sum.take_parts();
        let (acc, total) = self.chain(&sizes, &layout, parts, count)?;
        sum.set_folded(acc, total);
        Ok(())
    }

    /// The rank-ordered fold of every rank's `parts` (vars of `sizes` elements); returns the
    /// total and the total count on every rank.
    fn chain(&mut self, sizes: &[usize], layout: &str, parts: Vec<Part>, count: usize) -> Result<(Part, usize)> {
        let (r, w) = (self.rank(), self.world_size());
        let n = sizes.len();
        if let Some(p) = parts.iter().find(|p| p.len() != n || p.iter().zip(sizes).any(|(v, s)| v.as_ref().is_some_and(|v| v.len() != *s))) {
            return Err(NnError::Dist(format!("a contribution of {} vars does not match the layout ({n} vars)", p.len())));
        }
        if w == 1 || r == 0 {
            let mut acc: Part = vec![None; n];
            for (i, p) in parts.iter().enumerate() {
                fold(&mut acc, p, i > 0);
            }
            if w == 1 {
                return Ok((acc, count));
            }
            let seq = self.next_seq();
            let plan = plan(sizes, self.opts.chunk_elems);
            let mut e = Enc::default();
            e.str(layout).u64(count as u64).u64(plan.len() as u64).u64(self.opts.chunk_elems as u64);
            wire::send(self.peer(1), Op::ReduceMeta, 0, seq, &e.0)?;
            for (c, segs) in plan.iter().enumerate() {
                let msg = encode_chunk(c, segs, &acc);
                wire::send(self.peer(1), Op::ReduceChunk, 0, seq, &msg)?;
            }
            drop(acc);
            return self.recv_result(seq, sizes, &plan);
        }
        let seq = self.next_seq();
        let plan = plan(sizes, self.opts.chunk_elems);
        let payload = wire::recv(self.peer(r - 1), Op::ReduceMeta, r, r - 1, seq)?;
        let mut d = Dec::new(&payload);
        let (their_layout, before, chunks, chunk_elems) = (d.str()?, d.u64()? as usize, d.u64()? as usize, d.u64()? as usize);
        d.done()?;
        if their_layout != layout || chunks != plan.len() || chunk_elems != self.opts.chunk_elems {
            return Err(NnError::Dist(format!(
                "rank {r}: rank {} reduces another layout ({their_layout:?}, {chunks} chunks of {chunk_elems}; here {layout:?}, {} of {})",
                r - 1,
                plan.len(),
                self.opts.chunk_elems
            )));
        }
        let total_count = before + count;
        let last = r == w - 1;
        if !last {
            let mut e = Enc::default();
            e.str(layout).u64(total_count as u64).u64(plan.len() as u64).u64(self.opts.chunk_elems as u64);
            wire::send(self.peer(r + 1), Op::ReduceMeta, r, seq, &e.0)?;
        }
        let mut total: Part = vec![None; n];
        for (c, segs) in plan.iter().enumerate() {
            let payload = wire::recv(self.peer(r - 1), Op::ReduceChunk, r, r - 1, seq)?;
            let mut vals = decode_chunk(c, segs, &payload)?;
            // This rank's micro-steps, in order, after the ranks before it.
            for (i, p) in parts.iter().enumerate() {
                for (k, &(v, lo, hi)) in segs.iter().enumerate() {
                    fold_var(&mut vals[k], p[v].as_ref().map(|x| &x[lo..hi]), before + i > 0);
                }
            }
            if last {
                for (k, &(v, lo, hi)) in segs.iter().enumerate() {
                    if let Some(x) = vals[k].take() {
                        total[v].get_or_insert_with(|| vec![0.0; sizes[v]])[lo..hi].copy_from_slice(&x);
                    }
                }
            } else {
                let msg = encode_vals(c, &vals);
                wire::send(self.peer(r + 1), Op::ReduceChunk, r, seq, &msg)?;
            }
        }
        if !last {
            return self.recv_result(seq, sizes, &plan);
        }
        // The last rank holds the total: send it to every other rank, in rank order, once it has
        // all of it (so no rank is still sending while it waits for the result).
        let mut e = Enc::default();
        e.u64(total_count as u64).u64(plan.len() as u64);
        let meta = e.0;
        let msgs: Vec<Vec<u8>> = plan.iter().enumerate().map(|(c, segs)| encode_chunk(c, segs, &total)).collect();
        for q in 0..w - 1 {
            wire::send(self.peer(q), Op::ResultMeta, r, seq, &meta)?;
            for m in &msgs {
                wire::send(self.peer(q), Op::ResultChunk, r, seq, m)?;
            }
        }
        Ok((total, total_count))
    }

    fn recv_result(&mut self, seq: u64, sizes: &[usize], plan: &[Chunk]) -> Result<(Part, usize)> {
        let (r, last) = (self.rank(), self.world_size() - 1);
        let payload = wire::recv(self.peer(last), Op::ResultMeta, r, last, seq)?;
        let mut d = Dec::new(&payload);
        let (count, chunks) = (d.u64()? as usize, d.u64()? as usize);
        d.done()?;
        if chunks != plan.len() {
            return Err(NnError::Dist(format!("rank {r}: a result in {chunks} chunks, {} expected", plan.len())));
        }
        let mut total: Part = vec![None; sizes.len()];
        for (c, segs) in plan.iter().enumerate() {
            let payload = wire::recv(self.peer(last), Op::ResultChunk, r, last, seq)?;
            let vals = decode_chunk(c, segs, &payload)?;
            for (x, &(v, lo, hi)) in vals.into_iter().zip(segs) {
                if let Some(x) = x {
                    total[v].get_or_insert_with(|| vec![0.0; sizes[v]])[lo..hi].copy_from_slice(&x);
                }
            }
        }
        Ok((total, count))
    }
}

#[cfg(test)]
pub(crate) fn plan_for_tests(sizes: &[usize], max: usize) -> Vec<Vec<(usize, usize, usize)>> {
    plan(sizes, max)
}
