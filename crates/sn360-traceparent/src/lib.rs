//! W3C Trace Context generation + parsing for outbound Gateway requests.
//!
//! The SN360 agents do NOT (yet) embed a full OpenTelemetry Rust SDK —
//! that's a heavyweight pull that we don't need for the first
//! milestone. What this module does is the minimum that makes
//! agent-side traces correlate with the server-side spans the
//! Gateway and downstream services already emit:
//!
//! 1. Generate a fresh, W3C-compliant `traceparent` header for every
//!    outbound HTTP/2 request to the Gateway. The Gateway extracts
//!    it via `otelhttp.NewHandler`, attaches it to the inbound
//!    server span, and stamps it onto every NATS publish; downstream
//!    consumers (e.g. `correlation-engine`) continue the same trace.
//!
//! 2. If a caller has already populated a `traceparent` (e.g. a
//!    future OTel SDK integration), respect it — we never overwrite
//!    an explicit value. That keeps this module forward-compatible
//!    with a richer integration without forcing a rewrite of every
//!    gateway-transport caller.
//!
//! The format is fixed at version `00` (the only one the spec
//! defines today). 16-byte trace IDs and 8-byte span IDs are drawn
//! from the operating-system CSPRNG via [`getrandom`], so this crate
//! stays self-contained (no `ring` / `rand` transport dependency) and
//! every agent gets identical, cryptographically-strong identifiers.
//!
//! Reference: <https://www.w3.org/TR/trace-context/#traceparent-header>

use std::fmt::Write as _;

/// HTTP header name for the W3C Trace Context header.
pub const TRACEPARENT_HEADER: &str = "traceparent";

/// Generate a fresh W3C `traceparent` header value.
///
/// Returns a string of the form
/// `00-<32-hex-trace-id>-<16-hex-span-id>-01`, where the trailing
/// `01` is the W3C `sampled` flag. The agent always emits sampled
/// traces — the server-side OTel sampler is the single source of
/// volume-control authority, and sending `00` here would defeat the
/// whole purpose of the header.
pub fn generate_traceparent() -> String {
    let mut trace_id = [0u8; 16];
    let mut span_id = [0u8; 8];
    // `getrandom::fill` only fails if the OS RNG itself fails, which
    // is unrecoverable on a security agent. Surface that as a panic
    // rather than silently emitting an all-zero traceparent (which
    // the W3C spec explicitly forbids).
    getrandom::fill(&mut trace_id)
        .expect("system RNG must be available to generate a traceparent");
    getrandom::fill(&mut span_id)
        .expect("system RNG must be available to generate a traceparent");
    format_traceparent(&trace_id, &span_id, 0x01)
}

/// The trace-id / span-id components parsed out of a W3C
/// `traceparent` header value.
///
/// Both fields are normalised to lower-case hex (the W3C spec
/// mandates lower-case on the wire, but [`parse_traceparent`] is
/// lenient on input and normalises so downstream correlation keys
/// compare equal regardless of a producer's casing).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceParts {
    /// 32-char lower-case hex trace id (16 bytes).
    pub trace_id: String,
    /// 16-char lower-case hex span id (8 bytes).
    pub span_id: String,
}

/// Parse a W3C `traceparent` header value into its trace-id and
/// span-id components.
///
/// Returns `None` when `value` is not a well-formed version-`00`
/// traceparent. The validation mirrors the structural invariants
/// [`generate_traceparent`] guarantees and the W3C spec mandates:
///
/// * exactly four `-`-separated fields,
/// * version field equal to `00` (the only version defined today),
/// * a 32-hex-char trace id and a 16-hex-char span id,
/// * neither id all-zero (W3C §3.2.2.3 forbids the zero value).
///
/// The trailing flags field is accepted as any 2-hex-char value and
/// is not returned — the correlation context only needs the trace /
/// span identity, not the sampling decision (the agent always emits
/// sampled traces; see [`generate_traceparent`]).
///
/// This is the canonical parser co-located with the canonical
/// generator so the two stay in lock-step: a `traceparent` produced
/// by [`generate_traceparent`] always round-trips through
/// `parse_traceparent`.
pub fn parse_traceparent(value: &str) -> Option<TraceParts> {
    let mut fields = value.split('-');
    let version = fields.next()?;
    let trace_id = fields.next()?;
    let span_id = fields.next()?;
    let flags = fields.next()?;
    // Reject any trailing `-`-separated segment: a version-00
    // traceparent has exactly four fields.
    if fields.next().is_some() {
        return None;
    }
    if version != "00" {
        return None;
    }
    if !is_hex_of_len(trace_id, 32) || !is_hex_of_len(span_id, 16) {
        return None;
    }
    // Flags is a 2-hex-char field; validate its shape but discard it.
    if !is_hex_of_len(flags, 2) {
        return None;
    }
    // W3C §3.2.2.3 — the all-zero trace / span ids are invalid.
    if trace_id.bytes().all(|b| b == b'0') || span_id.bytes().all(|b| b == b'0') {
        return None;
    }
    Some(TraceParts {
        trace_id: trace_id.to_ascii_lowercase(),
        span_id: span_id.to_ascii_lowercase(),
    })
}

/// `true` when `s` is exactly `len` ASCII-hex characters.
fn is_hex_of_len(s: &str, len: usize) -> bool {
    s.len() == len && s.bytes().all(|b| b.is_ascii_hexdigit())
}

/// Check whether the iterator of header names already contains a
/// `traceparent` (case-insensitive per RFC 7230 §3.2). Centralizing
/// the predicate here means future additions to the W3C Trace
/// Context family (e.g. `tracestate`) only need to extend this one
/// function — both the steady-state gateway path and the enrollment
/// handshake share the same policy.
pub fn has_traceparent<'a, I>(header_names: I) -> bool
where
    I: IntoIterator<Item = &'a str>,
{
    header_names
        .into_iter()
        .any(|name| name.eq_ignore_ascii_case(TRACEPARENT_HEADER))
}

/// Stamp a freshly-generated `traceparent` on `extra_headers` unless
/// the caller has already supplied one (case-insensitive match per
/// RFC 7230 §3.2). Returns `true` if a header was added.
///
/// This is the `Vec<(String, String)>` flavor used by the
/// steady-state gateway dispatch path. The enrollment handshake uses
/// [`has_traceparent`] directly because it works with borrowed
/// `&[(&str, &str)]` slices and an `http::Request::Builder` rather
/// than a vector.
pub fn stamp_if_absent(extra_headers: &mut Vec<(String, String)>) -> bool {
    if has_traceparent(extra_headers.iter().map(|(name, _)| name.as_str())) {
        return false;
    }
    extra_headers.push((TRACEPARENT_HEADER.to_string(), generate_traceparent()));
    true
}

fn format_traceparent(trace_id: &[u8; 16], span_id: &[u8; 8], flags: u8) -> String {
    // 2 + 1 + 32 + 1 + 16 + 1 + 2 = 55 bytes for any well-formed
    // version-00 W3C traceparent. Pre-sizing the buffer keeps the
    // hot path allocation-free.
    let mut out = String::with_capacity(55);
    out.push_str("00-");
    for b in trace_id {
        // `write!` against a `String` cannot fail; the unwrap is
        // therefore total.
        write!(out, "{b:02x}").expect("string write");
    }
    out.push('-');
    for b in span_id {
        write!(out, "{b:02x}").expect("string write");
    }
    write!(out, "-{flags:02x}").expect("string write");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `generate_traceparent` produces a well-formed value: the
    /// fixed `00` version prefix, a 32-hex trace id, a 16-hex
    /// span id, and the sampled flag. We don't assert specific
    /// bytes — the CSPRNG guarantees those are random — but we DO
    /// assert the structural invariants the W3C spec mandates.
    #[test]
    fn traceparent_is_well_formed() {
        let tp = generate_traceparent();
        let parts: Vec<&str> = tp.split('-').collect();
        assert_eq!(parts.len(), 4, "expected 4 dash-separated fields: {tp}");
        assert_eq!(parts[0], "00", "version must be 00");
        assert_eq!(parts[1].len(), 32, "trace-id must be 32 hex chars: {tp}");
        assert_eq!(parts[2].len(), 16, "span-id must be 16 hex chars: {tp}");
        assert_eq!(parts[3], "01", "flags must be 01 (sampled)");
        // W3C §3.2.2.3 mandates lowercase hex. Production code uses
        // `{b:02x}` which already produces lowercase — the test
        // enforces it so a future refactor that switches to `{b:02X}`
        // (uppercase) gets caught here rather than at the wire.
        assert!(
            parts[1].chars().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
            "trace-id must be lowercase hex per W3C §3.2.2.3: {tp}"
        );
        assert!(
            parts[2].chars().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
            "span-id must be lowercase hex per W3C §3.2.2.3: {tp}"
        );
    }

    /// W3C §3.2.2.3 — an all-zero trace id is invalid. The CSPRNG
    /// makes this astronomically unlikely; the assertion exists so
    /// a buggy refactor that hard-codes zeros (or breaks the RNG
    /// wiring) is caught immediately rather than silently emitting
    /// invalid traceparents in production.
    #[test]
    fn traceparent_trace_id_is_nonzero() {
        let tp = generate_traceparent();
        let trace_id = tp.split('-').nth(1).unwrap();
        assert_ne!(
            trace_id, "00000000000000000000000000000000",
            "trace-id must never be all-zero (W3C §3.2.2.3): {tp}"
        );
    }

    /// W3C §3.2.2.3 — an all-zero span id is invalid for the same
    /// reason.
    #[test]
    fn traceparent_span_id_is_nonzero() {
        let tp = generate_traceparent();
        let span_id = tp.split('-').nth(2).unwrap();
        assert_ne!(
            span_id, "0000000000000000",
            "span-id must never be all-zero (W3C §3.2.2.3): {tp}"
        );
    }

    /// Two successive calls must produce distinct trace ids — that's
    /// the whole point of stamping a fresh one per request. With
    /// 128 random bits the probability of collision per pair is
    /// ~2^-128, so a test that observes a collision is observing a
    /// bug, not a flake.
    #[test]
    fn traceparent_ids_are_unique_across_calls() {
        let a = generate_traceparent();
        let b = generate_traceparent();
        assert_ne!(a, b, "traceparent must be unique per call");
    }

    /// A `traceparent` produced by [`generate_traceparent`] must
    /// round-trip through [`parse_traceparent`]: the canonical
    /// generator and canonical parser stay in lock-step.
    #[test]
    fn generated_traceparent_round_trips_through_parse() {
        let tp = generate_traceparent();
        let parts = parse_traceparent(&tp).expect("generated traceparent must parse");
        assert_eq!(parts.trace_id.len(), 32);
        assert_eq!(parts.span_id.len(), 16);
        // Re-formatting the parsed identity (with the sampled flag)
        // reproduces the original string.
        assert_eq!(format!("00-{}-{}-01", parts.trace_id, parts.span_id), tp);
    }

    /// `parse_traceparent` accepts a canonical W3C example and
    /// normalises uppercase input to lowercase.
    #[test]
    fn parse_accepts_canonical_and_normalises_case() {
        let preset = "00-0AF7651916CD43DD8448EB211C80319C-B7AD6B7169203331-01";
        let parts = parse_traceparent(preset).expect("canonical traceparent must parse");
        assert_eq!(parts.trace_id, "0af7651916cd43dd8448eb211c80319c");
        assert_eq!(parts.span_id, "b7ad6b7169203331");
    }

    /// `parse_traceparent` rejects malformed inputs: wrong field
    /// count, wrong version, bad lengths, non-hex, all-zero ids, and
    /// trailing segments.
    #[test]
    fn parse_rejects_malformed_inputs() {
        // Too few fields.
        assert!(parse_traceparent("00-abc-def").is_none());
        // Trailing extra segment.
        assert!(parse_traceparent(
            "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01-extra"
        )
        .is_none());
        // Unsupported version.
        assert!(parse_traceparent(
            "01-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
        )
        .is_none());
        // Trace-id wrong length.
        assert!(parse_traceparent("00-0af7651916cd43dd-b7ad6b7169203331-01").is_none());
        // Span-id non-hex.
        assert!(parse_traceparent(
            "00-0af7651916cd43dd8448eb211c80319c-zzzzzzzzzzzzzzzz-01"
        )
        .is_none());
        // All-zero trace id.
        assert!(parse_traceparent(
            "00-00000000000000000000000000000000-b7ad6b7169203331-01"
        )
        .is_none());
        // All-zero span id.
        assert!(parse_traceparent(
            "00-0af7651916cd43dd8448eb211c80319c-0000000000000000-01"
        )
        .is_none());
        // Flags wrong length.
        assert!(parse_traceparent(
            "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-001"
        )
        .is_none());
    }

    #[test]
    fn stamp_if_absent_adds_when_missing() {
        let mut headers = vec![("content-type".to_string(), "application/json".to_string())];
        assert!(stamp_if_absent(&mut headers));
        let tp = headers
            .iter()
            .find(|(n, _)| n.eq_ignore_ascii_case(TRACEPARENT_HEADER))
            .expect("traceparent should be stamped");
        assert!(tp.1.starts_with("00-"));
    }

    /// Idempotence: if the caller already provided a `traceparent`,
    /// `stamp_if_absent` must not overwrite it. This is what lets a
    /// future OTel-SDK integration coexist with this minimal
    /// stamper without a rewrite of every gateway call site.
    #[test]
    fn stamp_if_absent_does_not_overwrite_existing() {
        let preset = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01";
        let mut headers = vec![("traceparent".to_string(), preset.to_string())];
        assert!(!stamp_if_absent(&mut headers));
        assert_eq!(headers.len(), 1);
        assert_eq!(headers[0].1, preset);
    }

    /// HTTP header names are case-insensitive; a caller-supplied
    /// `Traceparent` (capital T) must still suppress the auto-stamp.
    #[test]
    fn stamp_if_absent_respects_case_insensitive_match() {
        let preset = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01";
        let mut headers = vec![("Traceparent".to_string(), preset.to_string())];
        assert!(!stamp_if_absent(&mut headers));
        assert_eq!(headers.len(), 1);
        assert_eq!(headers[0].1, preset);
    }

    #[test]
    fn format_traceparent_zero_pads_bytes() {
        let trace_id = [0x01u8; 16];
        let span_id = [0x02u8; 8];
        let tp = format_traceparent(&trace_id, &span_id, 0x01);
        assert_eq!(
            tp,
            "00-01010101010101010101010101010101-0202020202020202-01"
        );
    }

    /// `has_traceparent` is the shared policy predicate consumed
    /// by both `stamp_if_absent` (vector-of-owned-strings shape)
    /// and the enrollment path (slice-of-borrowed-strs shape).
    /// Verify it handles both case variants and unrelated headers.
    #[test]
    fn has_traceparent_detects_case_variants() {
        assert!(has_traceparent(["traceparent"]));
        assert!(has_traceparent(["Traceparent"]));
        assert!(has_traceparent(["TRACEPARENT"]));
        assert!(has_traceparent(["x-other", "traceparent"]));
        assert!(!has_traceparent(["x-other", "x-trace-id"]));
        assert!(!has_traceparent(std::iter::empty()));
    }
}
