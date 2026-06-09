//! Load PEM-encoded CA bundles into a rustls [`RootCertStore`].
//!
//! Every SN360 agent that pins a private trust anchor (rather than
//! relying on the public webpki roots) needs the same narrow
//! behaviour: parse a PEM bundle, trust *only* those certificates,
//! and reject an empty bundle up front so a misconfiguration fails
//! loud instead of fail-open. This crate is the single shared
//! implementation of that behaviour.
//!
//! ## Why not `rustls-pemfile`
//!
//! PEM is parsed via `rustls-pki-types`' typed [`PemObject`] API
//! rather than the `rustls-pemfile` 2.x crate, which is flagged
//! unmaintained by **RUSTSEC-2025-0134**. The semantics are
//! identical (iterate the `CERTIFICATE` PEM blocks, DER-decode
//! each), but the supported API keeps the agents off the advisory.
//!
//! ## Strict-pin semantics
//!
//! The returned [`RootCertStore`] contains *only* the certificates
//! in the supplied bundle — the webpki / platform root store is
//! **not** merged in. A caller that wants the public roots instead
//! should not call this loader at all and should fall back to its
//! own webpki path. There is deliberately no "additive" mode: the
//! agents reason about one root-trust knob, not per-call-site rules.

use std::path::Path;

use rustls::RootCertStore;
use rustls_pki_types::pem::PemObject;
use rustls_pki_types::CertificateDer;

/// Errors produced when loading a CA bundle.
#[derive(Debug, thiserror::Error)]
pub enum CaBundleError {
    /// The PEM file could not be read from disk. Only produced by
    /// [`load_ca_bundle_from_path`]; the byte-slice
    /// [`load_ca_bundle`] never performs IO.
    #[error("read ca bundle: {0}")]
    Io(#[from] std::io::Error),
    /// A certificate in the PEM stream failed to parse.
    #[error("ca pem parse: {0}")]
    Parse(String),
    /// `RootCertStore::add` rejected a certificate (typically a
    /// malformed X.509 body).
    #[error("ca add: {0}")]
    Add(String),
    /// The PEM input parsed cleanly but contained no certificates —
    /// an empty trust store would silently fail every handshake, so
    /// we treat this as a configuration error up front.
    #[error("ca bundle contained no certificates")]
    Empty,
}

/// Load a PEM-encoded CA bundle from a byte slice into a fresh
/// [`RootCertStore`].
///
/// This is the canonical entry point. Only the certificates in
/// `pem_bytes` are trusted; the webpki / platform root store is
/// **not** merged in (see the module docs on strict-pin semantics).
///
/// Returns:
/// * [`CaBundleError::Parse`] if a PEM `CERTIFICATE` block fails to
///   DER-decode,
/// * [`CaBundleError::Add`] if rustls rejects an otherwise-decoded
///   certificate,
/// * [`CaBundleError::Empty`] if the input carried zero
///   certificates (including non-PEM garbage, which yields no
///   `CERTIFICATE` blocks).
pub fn load_ca_bundle(pem_bytes: &[u8]) -> Result<RootCertStore, CaBundleError> {
    let mut roots = RootCertStore::empty();
    let mut added = 0usize;
    for cert in CertificateDer::pem_slice_iter(pem_bytes) {
        let cert = cert.map_err(|e| CaBundleError::Parse(e.to_string()))?;
        roots
            .add(cert)
            .map_err(|e| CaBundleError::Add(e.to_string()))?;
        added += 1;
    }
    if added == 0 {
        return Err(CaBundleError::Empty);
    }
    Ok(roots)
}

/// Read a PEM-encoded CA bundle from disk and load it via
/// [`load_ca_bundle`].
///
/// A thin convenience wrapper for the common "operator points at a
/// `ca_bundle` file on disk" path. The empty-bundle, parse-error and
/// strict-pin semantics are exactly those of [`load_ca_bundle`]; the
/// only additional failure mode is [`CaBundleError::Io`] when the
/// file cannot be read.
pub fn load_ca_bundle_from_path(pem_path: &Path) -> Result<RootCertStore, CaBundleError> {
    let bytes = std::fs::read(pem_path)?;
    load_ca_bundle(&bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn rejects_empty_byte_slice() {
        let err = load_ca_bundle(b"").expect_err("empty input must error");
        assert!(matches!(err, CaBundleError::Empty), "got {err:?}");
    }

    #[test]
    fn rejects_non_pem_garbage() {
        // Garbage parses as "no certificates found" — the empty-store
        // guard upgrades that to a typed error rather than returning a
        // fail-open empty trust store.
        let err = load_ca_bundle(b"not a pem file, just some bytes\n")
            .expect_err("garbage must error");
        assert!(matches!(err, CaBundleError::Empty), "got {err:?}");
    }

    #[test]
    fn rejects_missing_file() {
        let err = load_ca_bundle_from_path(Path::new(
            "/nonexistent/sn360-tls-roots-loader-test.pem.does-not-exist",
        ))
        .expect_err("missing file must error");
        assert!(matches!(err, CaBundleError::Io(_)), "got {err:?}");
    }

    #[test]
    fn rejects_empty_file() {
        let tmp = tempfile::NamedTempFile::new().expect("tempfile");
        let err = load_ca_bundle_from_path(tmp.path()).expect_err("empty bundle must error");
        assert!(matches!(err, CaBundleError::Empty), "got {err:?}");
    }

    /// Full positive round-trip: generate a self-signed cert via
    /// `rcgen`, load it both from bytes and from a tempfile, and
    /// verify the resulting trust store is non-empty. Validates the
    /// happy path end-to-end without depending on a pre-baked PEM
    /// fixture.
    #[test]
    fn loads_a_self_signed_bundle() {
        let cert = rcgen::generate_simple_self_signed(vec![
            "sn360-tls-roots-test.example.com".to_string(),
        ])
        .expect("generate cert");
        let pem = cert.cert.pem();

        // Byte-slice loader.
        let roots = load_ca_bundle(pem.as_bytes()).expect("load ca bundle from bytes");
        assert!(
            !roots.is_empty(),
            "byte loader must produce a non-empty trust store on a valid PEM"
        );

        // Path loader.
        let mut tmp = tempfile::NamedTempFile::new().expect("tempfile");
        tmp.write_all(pem.as_bytes()).expect("write pem");
        tmp.flush().expect("flush");
        let roots = load_ca_bundle_from_path(tmp.path()).expect("load ca bundle from path");
        assert_eq!(
            roots.len(),
            1,
            "a single-cert bundle must yield exactly one trust anchor (no implicit webpki merge)"
        );
    }
}
