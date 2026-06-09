//! Shared POD wire types for the SN360 agent eBPF pipeline.
//!
//! This crate is intentionally `#![no_std]` and dependency-free so
//! the exact same source compiles for both halves of the eBPF
//! event pipeline in every SN360 agent (desktop and VM today; the
//! K8s agent can opt in later via the `k8s` feature):
//!
//! * The **kernel-side** eBPF crate (compiled for
//!   `bpfel-unknown-none`, `no_std` + `no_main`, fed to bpf-linker).
//! * The **userland** loader crate (compiled for the host's
//!   regular target). The aya ring-buffer reader pulls raw bytes
//!   off the kernel-allocated map and casts them to the variant
//!   indicated by the [`WireKind`] discriminator in the header.
//!
//! ## Wire format
//!
//! Every ring-buffer record starts with a fixed-size [`WireHeader`]
//! that carries a [`WireKind`] discriminator. The discriminator does
//! NOT sit at byte 0: the header is laid out to keep `ktime_ns`
//! 8-byte aligned at offset 0, so the kind byte lands at
//! [`WIRE_KIND_OFFSET`] (after the three 4-byte identifier fields).
//! The userland reader reads the kind from that offset (see
//! [`userland::peek_kind`]), then casts the whole record to the
//! `#[repr(C)]` struct sized to the variant.
//!
//! All fields are little-endian (which matches the bpfel target
//! and every Linux platform the SN360 agents support — the bpfeb
//! target is intentionally NOT a supported deployment target).
//! Multi-byte integers MUST use the `#[repr(C)]` layout produced by
//! Rust's struct codegen so the kernel and userland views agree
//! byte-for-byte; we never serialize via `to_le_bytes()` in the
//! kernel path because that would force the kernel programs onto a
//! BPF stack budget they cannot afford.
//!
//! ## Stack / map sizing constraints
//!
//! The BPF verifier limits each program's stack frame to 512 bytes.
//! Variants that would exceed that budget on a single stack frame
//! (notably [`WireExec`], which carries both an `exe` and a
//! `cmdline` buffer) MUST be assembled inside a `BPF_MAP_TYPE_PERCPU_ARRAY`
//! scratch slot instead of being constructed on the local stack
//! and copied into a ring-buffer reservation. Each agent's
//! kernel-side crate enforces this pattern.

#![no_std]
#![forbid(unsafe_op_in_unsafe_fn)]
#![deny(missing_docs)]

/// Variant discriminator carried in every ring-buffer record's
/// [`WireHeader`]. It lives at [`WIRE_KIND_OFFSET`] (offset 20), NOT
/// at byte 0 — read it via [`userland::peek_kind`] rather than the
/// leading byte.
///
/// The numeric values are part of the on-the-wire contract — bumping
/// them breaks the build pipeline because the userland decoder and
/// the kernel programs would disagree on which struct to cast a
/// record to. New variants append at the end with a fresh integer.
#[repr(u8)]
#[derive(Copy, Clone)]
#[cfg_attr(feature = "user", derive(Debug, PartialEq, Eq))]
pub enum WireKind {
    /// `WireExec` — `sched_process_exec` or `sys_enter_execve`
    /// payload.
    Exec = 1,
    /// `WireExit` — `sched_process_exit` payload.
    Exit = 2,
    /// `WireOpen` — `sys_enter_openat` payload.
    OpenFile = 3,
    /// `WireNet` — `tcp_connect` / `inet_csk_accept` payload.
    Network = 4,
    /// `WireDns` — `udp_sendmsg` on UDP/53 payload.
    Dns = 5,
}

impl WireKind {
    /// Convert a raw byte read off the ring buffer back into a
    /// [`WireKind`]. Returns `None` for unknown discriminators so
    /// the userland reader can log + skip rather than aborting the
    /// task on a single malformed record.
    pub const fn from_u8(b: u8) -> Option<Self> {
        match b {
            1 => Some(Self::Exec),
            2 => Some(Self::Exit),
            3 => Some(Self::OpenFile),
            4 => Some(Self::Network),
            5 => Some(Self::Dns),
            _ => None,
        }
    }
}

/// Maximum length (bytes, including the trailing NUL byte slot we
/// don't actually store) of a captured executable path. Sized to
/// cover every realistic Linux binary path without breaking the
/// BPF stack budget — `/usr/lib/<distro>/...` paths comfortably fit
/// in 256.
pub const EXE_LEN: usize = 256;

/// Maximum length (bytes) of the captured command line. Per-arg we
/// only read up to `EXE_LEN`, then concatenate space-separated up
/// to this cap. 512 bytes captures the vast majority of legitimate
/// service / cron / package-manager invocations; longer cmdlines
/// (large `--exec` payloads, command-injection attempts that flood
/// argv) are truncated and a `\x00` terminator is appended.
pub const CMDLINE_LEN: usize = 512;

/// Maximum length (bytes) of the captured current working
/// directory.
pub const CWD_LEN: usize = 256;

/// Maximum length (bytes) of the captured open() path. Mirrors
/// `EXE_LEN` so the verifier sees a single uniform `bpf_probe_read_user_str`
/// size class for every string field.
pub const PATH_LEN: usize = 256;

/// Maximum length (bytes) of the captured DNS query.
pub const DNS_QUERY_LEN: usize = 256;

/// Common header carried at the start of every wire variant.
///
/// Field order is chosen so the resulting `#[repr(C)]` struct is
/// exactly 24 bytes with no implicit padding inserted by the
/// compiler: the `u64` ktime field comes first at offset 0 (8-byte
/// aligned), followed by three `u32`s (offsets 8/12/16), then the
/// 1-byte kind discriminator at offset 20 with 3 bytes of trailing
/// padding. The kind byte does NOT sit at offset 0 — the userland
/// decoder reads it from offset 20 via [`userland::peek_kind`] (which knows
/// the layout), and the kernel programs assemble the header by
/// writing field-by-field so the same convention holds.
#[repr(C)]
#[derive(Copy, Clone)]
#[cfg_attr(feature = "user", derive(Debug, PartialEq, Eq))]
pub struct WireHeader {
    /// Monotonic ktime (nanoseconds since boot) captured by
    /// `bpf_ktime_get_ns()` on the kernel side.
    pub ktime_ns: u64,
    /// PID of the originating task.
    pub pid: u32,
    /// Parent PID (best-effort — from a kernel-maintained
    /// fork-tracking map). `0` means "unknown" / "tracker not yet
    /// seeded" and userland callers should treat it as a missing
    /// value rather than a real init-child.
    pub ppid: u32,
    /// EUID of the originating task.
    pub uid: u32,
    /// Variant discriminator; cast back to [`WireKind`] on the
    /// userland side via [`userland::peek_kind`].
    pub kind: u8,
    /// Reserved padding so the struct size is a multiple of the
    /// 8-byte alignment requirement.
    pub _pad: [u8; 3],
}

/// Byte offset of [`WireHeader::kind`] within the header. The
/// userland decoder's [`userland::peek_kind`] uses this constant rather than
/// reading byte 0; bumping the header layout requires touching
/// both sides of the contract.
pub const WIRE_KIND_OFFSET: usize = 20;

/// `WireExec` — emitted on `sys_enter_execve` (or `sched_process_exec`,
/// depending on the bundled kernel object).
///
/// The kernel program assembles this struct inside a per-CPU
/// scratch slot to avoid blowing the BPF stack budget. The reader
/// on the userland side trims `exe` / `cmdline` / `cwd` at the
/// first NUL byte.
#[repr(C)]
#[derive(Copy, Clone)]
#[cfg_attr(feature = "user", derive(Debug, PartialEq, Eq))]
pub struct WireExec {
    /// Common header.
    pub header: WireHeader,
    /// NUL-terminated executable path (best-effort — from the
    /// tracepoint's `filename` argument).
    pub exe: [u8; EXE_LEN],
    /// NUL-terminated space-joined command line. Truncated at
    /// [`CMDLINE_LEN`].
    pub cmdline: [u8; CMDLINE_LEN],
    /// NUL-terminated current working directory. Empty if the
    /// kernel did not surface it.
    pub cwd: [u8; CWD_LEN],
}

/// `WireExit` — emitted on `sched_process_exit`.
#[repr(C)]
#[derive(Copy, Clone)]
#[cfg_attr(feature = "user", derive(Debug, PartialEq, Eq))]
pub struct WireExit {
    /// Common header.
    pub header: WireHeader,
    /// `wait(2)`-style exit status. The BPF `sched_process_exit`
    /// tracepoint does not surface the exit code directly; the
    /// kernel program reads it from `task->exit_code` via a
    /// `bpf_probe_read_kernel` against the offset baked in at
    /// build time. When the read fails it surfaces `0` rather than
    /// guessing — userland callers should treat `0` as "unknown"
    /// for the rule-side path (legitimate exit codes are
    /// indistinguishable on the wire from a read failure, by
    /// design).
    pub exit_code: i32,
    /// Reserved so the struct size is a multiple of 8.
    pub _pad: [u8; 4],
}

/// `WireOpen` — emitted on `sys_enter_openat`.
///
/// File-open events flow to the agent's FIM rule pipeline via the
/// kernel probe's own subscription path, NOT through the
/// `EventBus`. The userland loader's `ebpf_event_to_event_kind`
/// returns `None` for this variant.
#[repr(C)]
#[derive(Copy, Clone)]
#[cfg_attr(feature = "user", derive(Debug, PartialEq, Eq))]
pub struct WireOpen {
    /// Common header.
    pub header: WireHeader,
    /// `open(2)` flags (`O_RDONLY` / `O_WRONLY` / `O_CREAT` …).
    pub flags: i32,
    /// Reserved so `path` starts at offset 32: the 24-byte header
    /// plus `flags` plus this padding make up a fixed 32-byte prefix,
    /// giving the struct the `24 + 8 + PATH_LEN` `#[repr(C)]` layout
    /// asserted below. `path` is a byte array (alignment 1), so this
    /// padding pins the wire offset rather than satisfying an
    /// alignment requirement.
    pub _pad: [u8; 4],
    /// NUL-terminated path argument to `openat(2)`.
    pub path: [u8; PATH_LEN],
}

/// `WireNet` — emitted on `tcp_connect` (outbound) and
/// `inet_csk_accept` (inbound).
///
/// Addresses are stored in network byte order to match the wire
/// representation kernel-side; userland renders them via the
/// standard `Ipv4Addr` / `Ipv6Addr` types so byte order never
/// becomes ambiguous outside this struct.
#[repr(C)]
#[derive(Copy, Clone)]
#[cfg_attr(feature = "user", derive(Debug, PartialEq, Eq))]
pub struct WireNet {
    /// Common header.
    pub header: WireHeader,
    /// Address family: `AF_INET` (2) or `AF_INET6` (10). Other
    /// families are dropped on the kernel side.
    pub family: u8,
    /// Direction: `1` outbound, `2` inbound.
    pub direction: u8,
    /// IP protocol: `IPPROTO_TCP` (6) or `IPPROTO_UDP` (17).
    pub protocol: u8,
    /// Reserved.
    pub _pad: u8,
    /// Source port (host byte order).
    pub src_port: u16,
    /// Destination port (host byte order).
    pub dst_port: u16,
    /// Source IP. For `AF_INET` only the first 4 bytes are
    /// meaningful; the rest is zero-padded.
    pub src_addr: [u8; 16],
    /// Destination IP. Same convention as [`Self::src_addr`].
    pub dst_addr: [u8; 16],
}

/// `WireDns` — emitted on `udp_sendmsg` when the destination port
/// is 53.
///
/// The kernel program copies up to [`DNS_QUERY_LEN`] bytes of the
/// raw UDP payload into [`Self::query`]; the userland decoder
/// applies the DNS message format (header + qname label parsing)
/// to extract the actual qname / qtype, so this field is
/// intentionally NUL-terminated raw bytes, NOT a pre-parsed string.
#[repr(C)]
#[derive(Copy, Clone)]
#[cfg_attr(feature = "user", derive(Debug, PartialEq, Eq))]
pub struct WireDns {
    /// Common header.
    pub header: WireHeader,
    /// Length (bytes) of the captured UDP payload. May be less than
    /// [`DNS_QUERY_LEN`] when the actual datagram was shorter.
    pub query_len: u16,
    /// Reserved.
    pub _pad: [u8; 6],
    /// Raw UDP payload (DNS message); userland-side parses the
    /// DNS header + qname out of this buffer. Bounded by
    /// [`DNS_QUERY_LEN`] so the kernel program's
    /// `bpf_probe_read_user` call has a verifier-friendly size
    /// constant.
    pub query: [u8; DNS_QUERY_LEN],
}

/// Compile-time assertion that [`WireHeader`] has the documented
/// 24-byte layout. Catches an accidental field reorder before the
/// kernel object is compiled.
const _: () = assert!(core::mem::size_of::<WireHeader>() == 24);
/// Same shape check for [`WireExit`].
const _: () = assert!(core::mem::size_of::<WireExit>() == 24 + 8);
/// Same shape check for [`WireOpen`].
const _: () = assert!(core::mem::size_of::<WireOpen>() == 24 + 8 + PATH_LEN);
/// Same shape check for [`WireNet`].
const _: () = assert!(core::mem::size_of::<WireNet>() == 24 + 8 + 32);
/// Same shape check for [`WireDns`].
const _: () = assert!(core::mem::size_of::<WireDns>() == 24 + 8 + DNS_QUERY_LEN);
/// Same shape check for [`WireExec`] — sanity bound on the worst-
/// case ring-buffer record. Catches a regression that pushes the
/// per-record cost above the loader's `RECORD_SIZE_GUARD`.
const _: () = assert!(core::mem::size_of::<WireExec>() == 24 + EXE_LEN + CMDLINE_LEN + CWD_LEN);

mod sealed {
    /// Sealing supertrait for [`super::WireRecord`]. Keeping it private
    /// prevents downstream crates from implementing `WireRecord` for
    /// their own types, so the set of byte-castable types stays closed
    /// to the verified POD layouts defined in this crate.
    pub trait Sealed {}
}

/// Fixed-layout wire records that are safe to materialize from a raw
/// ring-buffer byte slice via [`userland::cast_record`].
///
/// The trait is sealed: only the `#[repr(C)]` `Wire*` structs defined
/// in this crate implement it. Each is composed solely of integer
/// primitives and `[u8; N]` byte arrays, so every bit pattern of the
/// correct length is a valid value — the invariant `cast_record`'s
/// `read_unaligned` relies on for soundness. Because the trait cannot
/// be implemented downstream, a consumer cannot widen `cast_record` to
/// a type with invalid bit patterns (e.g. `bool` or a
/// restricted-discriminant enum), which would otherwise be UB.
pub trait WireRecord: sealed::Sealed + Copy {}

macro_rules! impl_wire_record {
    ($($t:ty),+ $(,)?) => {
        $(
            impl sealed::Sealed for $t {}
            impl WireRecord for $t {}
        )+
    };
}

impl_wire_record!(WireHeader, WireExec, WireExit, WireOpen, WireNet, WireDns);

/// Userland helpers — only compiled when the consumer (the agent's
/// userland eBPF loader crate) opts in via the `user` feature.
/// Keeping them gated keeps `core::fmt` panics out of the
/// kernel ELF.
#[cfg(feature = "user")]
pub mod userland {
    use super::*;

    /// Trim a fixed-size NUL-terminated byte buffer at the first
    /// `\0`. Returns the resulting `str` if the prefix is valid
    /// UTF-8; otherwise returns an empty string. Both branches are
    /// kept lossless — callers that need to surface non-UTF-8
    /// kernel-side strings can borrow the raw `&[u8]` instead.
    pub fn cstr_lossy(buf: &[u8]) -> &str {
        let end = buf.iter().position(|b| *b == 0).unwrap_or(buf.len());
        core::str::from_utf8(&buf[..end]).unwrap_or("")
    }

    /// Decode the [`WireKind`] discriminator off a raw ring-buffer
    /// record. Returns `None` if the record is shorter than the
    /// header layout requires or carries an unknown discriminator.
    ///
    /// The discriminator lives at [`WIRE_KIND_OFFSET`] inside the
    /// [`WireHeader`], NOT at byte 0 — the header is laid out to
    /// keep `ktime_ns` 8-byte aligned at offset 0, so the kind
    /// byte sits after the three 4-byte process-identifier fields.
    pub fn peek_kind(bytes: &[u8]) -> Option<WireKind> {
        bytes
            .get(WIRE_KIND_OFFSET)
            .copied()
            .and_then(WireKind::from_u8)
    }

    /// Cast a raw ring-buffer record to a `T` of the matching wire
    /// variant. Returns `None` when the buffer is shorter than `T` —
    /// the caller is responsible for matching `T` to the [`WireKind`]
    /// read out of [`WIRE_KIND_OFFSET`].
    ///
    /// `T` is bounded by the sealed [`WireRecord`] trait, so it can
    /// only ever be one of this crate's `#[repr(C)]` POD wire structs
    /// (every byte pattern of the right length is a valid value). The
    /// unaligned read below is therefore sound for any sufficiently
    /// long slice, and downstream code cannot instantiate it with a
    /// type that has invalid bit patterns.
    pub fn cast_record<T: WireRecord>(bytes: &[u8]) -> Option<T> {
        if bytes.len() < core::mem::size_of::<T>() {
            return None;
        }
        // SAFETY: `T: WireRecord` is `#[repr(C)]` POD. We perform an unaligned
        // read because the kernel-allocated ring buffer may yield
        // records at arbitrary byte offsets. `read_unaligned` is
        // the documented sound way to materialize a POD from a
        // possibly-unaligned source pointer in `core`.
        let value = unsafe { core::ptr::read_unaligned(bytes.as_ptr().cast::<T>()) };
        Some(value)
    }
}

#[cfg(all(test, feature = "user"))]
mod tests {
    use super::*;

    #[test]
    fn wire_kind_round_trips_through_u8() {
        for v in [
            WireKind::Exec,
            WireKind::Exit,
            WireKind::OpenFile,
            WireKind::Network,
            WireKind::Dns,
        ] {
            let raw = v as u8;
            assert_eq!(WireKind::from_u8(raw), Some(v));
        }
        assert_eq!(WireKind::from_u8(0), None);
        assert_eq!(WireKind::from_u8(99), None);
    }

    #[test]
    fn cstr_lossy_trims_at_first_nul() {
        let mut buf = [0u8; 16];
        buf[..5].copy_from_slice(b"hello");
        assert_eq!(userland::cstr_lossy(&buf), "hello");
    }

    #[test]
    fn cstr_lossy_handles_full_buffer() {
        let buf = [b'a'; 8];
        assert_eq!(userland::cstr_lossy(&buf), "aaaaaaaa");
    }

    #[test]
    fn cstr_lossy_returns_empty_on_invalid_utf8() {
        let buf = [0xFF, 0xFE, 0, 0, 0, 0, 0, 0];
        assert_eq!(userland::cstr_lossy(&buf), "");
    }

    #[test]
    fn cast_record_rejects_short_buffer() {
        let bytes = [0u8; 8];
        assert!(userland::cast_record::<WireExit>(&bytes).is_none());
    }

    #[test]
    fn cast_record_reads_unaligned() {
        // Construct a `WireExit` value, copy its bytes into a
        // buffer with a 1-byte offset, and verify the unaligned
        // read still recovers the original.
        let original = WireExit {
            header: WireHeader {
                kind: WireKind::Exit as u8,
                _pad: [0; 3],
                ktime_ns: 0x0102_0304_0506_0708,
                pid: 1234,
                ppid: 1,
                uid: 1000,
            },
            exit_code: -42,
            _pad: [0; 4],
        };
        let mut storage = [0u8; 1 + core::mem::size_of::<WireExit>()];
        // SAFETY: `WireExit` is `#[repr(C)]` POD; we materialize
        // its little-endian byte layout for the test fixture.
        let raw: [u8; core::mem::size_of::<WireExit>()] =
            unsafe { core::mem::transmute_copy(&original) };
        storage[1..].copy_from_slice(&raw);

        let decoded = userland::cast_record::<WireExit>(&storage[1..]).expect("decode at offset 1");
        assert_eq!(decoded, original);
    }
}
