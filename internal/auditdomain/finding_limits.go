package auditdomain

// Wire limits for finding-proposal.v1. Their rationale and Python counterparts
// are specified in docs/spec/27-findings-tools-and-collections.md.
const (
	MaximumFindingLocations     = MaximumEvidencePerItem
	MaximumFindingPathBytes     = 4096      // V63-001 relative POSIX path contract.
	MaximumFindingURLBytes      = 8192      // Existing http-tools MAX_URL_BYTES.
	MaximumFindingMethodBytes   = 64        // V63-001 HTTP token contract.
	MaximumFindingLine          = 1<<53 - 1 // Largest exactly representable JSON/JS integer.
	MaximumHTTPHeaderBytes      = 64 << 10  // Existing http-tools MAX_HEADER_BYTES.
	MaximumHTTPRequestBodyBytes = 1 << 20   // Existing http-tools MAX_REQUEST_BODY_BYTES.
	MaximumHTTPRedirects        = 10        // Existing http-tools MAX_REDIRECTS.
	MaximumHTTPAttempts         = 3         // Existing http-tools MAX_ATTEMPTS; shared retry budget.
	MaximumHTTPExchangeAttempts = MaximumHTTPRedirects + MaximumHTTPAttempts
)
