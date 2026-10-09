//! The wire format of the rendezvous and the collectives. Every message is a header (a magic word,
//! the operation, the sender's rank, the collective's sequence number, the payload's length) and
//! the payload. A receiver checks the operation, the sender and the sequence number, so ranks
//! that call different collectives, or the same ones in another order, fail at once instead of
//! mixing data. Numbers are little-endian.

use crate::error::{NnError, Result};
use std::io::{ErrorKind, Read, Write};
use std::net::TcpStream;

const MAGIC: [u8; 4] = *b"tcol";
const HEADER: usize = 4 + 4 + 4 + 8 + 8;
/// The largest payload a receiver accepts (a guard against a corrupt length).
const MAX_PAYLOAD: u64 = 1 << 34;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub(crate) enum Op {
    Hello = 1,
    Table = 2,
    Link = 3,
    Barrier = 4,
    Broadcast = 5,
    Gather = 6,
    ReduceMeta = 7,
    ReduceChunk = 8,
    ResultMeta = 9,
    ResultChunk = 10,
    Challenge = 11,
    LinkAck = 12,
}

impl Op {
    fn name(self) -> &'static str {
        match self {
            Op::Hello => "hello",
            Op::Table => "table",
            Op::Link => "link",
            Op::Barrier => "barrier",
            Op::Broadcast => "broadcast",
            Op::Gather => "all-gather",
            Op::ReduceMeta | Op::ReduceChunk => "all-reduce",
            Op::ResultMeta | Op::ResultChunk => "all-reduce result",
            Op::Challenge => "challenge",
            Op::LinkAck => "link answer",
        }
    }

    fn from_u32(v: u32) -> Option<Op> {
        [Op::Hello, Op::Table, Op::Link, Op::Barrier, Op::Broadcast, Op::Gather, Op::ReduceMeta, Op::ReduceChunk, Op::ResultMeta, Op::ResultChunk, Op::Challenge, Op::LinkAck].into_iter().find(|o| *o as u32 == v)
    }
}

/// Send one message.
pub(crate) fn send(s: &mut TcpStream, op: Op, rank: usize, seq: u64, payload: &[u8]) -> Result<()> {
    let mut h = Vec::with_capacity(HEADER + payload.len().min(1 << 16));
    h.extend_from_slice(&MAGIC);
    h.extend_from_slice(&(op as u32).to_le_bytes());
    h.extend_from_slice(&(rank as u32).to_le_bytes());
    h.extend_from_slice(&seq.to_le_bytes());
    h.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    let io = |e: std::io::Error| NnError::Dist(format!("rank {rank}: sending {} #{seq}: {e} (the peer stopped?)", op.name()));
    if payload.len() <= 1 << 16 {
        h.extend_from_slice(payload);
        s.write_all(&h).map_err(io)?;
    } else {
        s.write_all(&h).map_err(io)?;
        s.write_all(payload).map_err(io)?;
    }
    Ok(())
}

/// Receive one message, which must be `op` #`seq` from rank `from`; `me` names the receiver in
/// errors.
pub(crate) fn recv(s: &mut TcpStream, op: Op, me: usize, from: usize, seq: u64) -> Result<Vec<u8>> {
    let (got, sender, got_seq, payload) = recv_any(s, me, from)?;
    if got != op || sender != from || got_seq != seq {
        return Err(NnError::Dist(format!(
            "rank {me}: expected {} #{seq} from rank {from}, got {} #{got_seq} from rank {sender} (the ranks called different collectives)",
            op.name(),
            got.name()
        )));
    }
    Ok(payload)
}

/// Receive one message of any kind: (operation, sender, sequence number, payload). `from` is the
/// expected peer, for errors only.
pub(crate) fn recv_any(s: &mut TcpStream, me: usize, from: usize) -> Result<(Op, usize, u64, Vec<u8>)> {
    let mut h = [0u8; HEADER];
    read_exact(s, &mut h, me, from)?;
    if h[0..4] != MAGIC {
        return Err(NnError::Dist(format!("rank {me}: a message from rank {from} without the collectives' magic word (another service on the port?)")));
    }
    let u32_at = |i: usize| u32::from_le_bytes(h[i..i + 4].try_into().expect("4 bytes"));
    let u64_at = |i: usize| u64::from_le_bytes(h[i..i + 8].try_into().expect("8 bytes"));
    let op = Op::from_u32(u32_at(4)).ok_or_else(|| NnError::Dist(format!("rank {me}: unknown operation {} from rank {from}", u32_at(4))))?;
    let (sender, seq, len) = (u32_at(8) as usize, u64_at(12), u64_at(20));
    if len > MAX_PAYLOAD {
        return Err(NnError::Dist(format!("rank {me}: a payload of {len} bytes from rank {from}")));
    }
    let mut payload = vec![0u8; len as usize];
    read_exact(s, &mut payload, me, from)?;
    Ok((op, sender, seq, payload))
}

fn read_exact(s: &mut TcpStream, buf: &mut [u8], me: usize, from: usize) -> Result<()> {
    s.read_exact(buf).map_err(|e| match e.kind() {
        ErrorKind::UnexpectedEof | ErrorKind::ConnectionReset | ErrorKind::ConnectionAborted | ErrorKind::BrokenPipe => {
            NnError::Dist(format!("rank {me}: rank {from} closed the connection (it stopped)"))
        }
        ErrorKind::WouldBlock | ErrorKind::TimedOut => NnError::Dist(format!("rank {me}: no message from rank {from} within the timeout")),
        _ => NnError::Dist(format!("rank {me}: reading from rank {from}: {e}")),
    })
}

/// A little-endian encoder for payloads.
#[derive(Default)]
pub(crate) struct Enc(pub Vec<u8>);

impl Enc {
    pub fn u8(&mut self, v: u8) -> &mut Self {
        self.0.push(v);
        self
    }
    pub fn u64(&mut self, v: u64) -> &mut Self {
        self.0.extend_from_slice(&v.to_le_bytes());
        self
    }
    pub fn bytes(&mut self, v: &[u8]) -> &mut Self {
        self.u64(v.len() as u64);
        self.0.extend_from_slice(v);
        self
    }
    pub fn str(&mut self, v: &str) -> &mut Self {
        self.bytes(v.as_bytes())
    }
    pub fn f32s(&mut self, v: &[f32]) -> &mut Self {
        self.0.reserve(4 * v.len());
        for x in v {
            self.0.extend_from_slice(&x.to_bits().to_le_bytes());
        }
        self
    }
}

/// The matching decoder; every read checks the length.
pub(crate) struct Dec<'a> {
    b: &'a [u8],
    at: usize,
}

impl<'a> Dec<'a> {
    pub fn new(b: &'a [u8]) -> Self {
        Dec { b, at: 0 }
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.b.len() - self.at < n {
            return Err(NnError::Dist(format!("a truncated message ({} bytes, {} more needed at {})", self.b.len(), n, self.at)));
        }
        let s = &self.b[self.at..self.at + n];
        self.at += n;
        Ok(s)
    }
    pub fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    pub fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().expect("8 bytes")))
    }
    pub fn bytes(&mut self) -> Result<&'a [u8]> {
        let n = self.u64()? as usize;
        self.take(n)
    }
    pub fn str(&mut self) -> Result<String> {
        String::from_utf8(self.bytes()?.to_vec()).map_err(|_| NnError::Dist("a message with malformed text".into()))
    }
    pub fn f32s(&mut self, n: usize) -> Result<Vec<f32>> {
        let b = self.take(4 * n)?;
        Ok(b.chunks_exact(4).map(|c| f32::from_bits(u32::from_le_bytes([c[0], c[1], c[2], c[3]]))).collect())
    }
    /// Bytes read so far.
    pub fn at(&self) -> usize {
        self.at
    }
    pub fn done(&self) -> Result<()> {
        if self.at != self.b.len() {
            return Err(NnError::Dist(format!("a message with {} trailing bytes", self.b.len() - self.at)));
        }
        Ok(())
    }
}
