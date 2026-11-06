package evalservice

// These pins identify the built-in policies' behavior. The V1 pins match the
// policies shipped before explicit versions were added. Change a pin only when
// its behavior changes intentionally; source edits, comments, and refactors
// must not invalidate existing Experiments.
const (
	humanPolicyV1SHA256 = "sha256:25740d3d9f24f4894b4f44d4e90585df006712bc84017be40114c6c1fea6d8c1"
	// Version 2 replaces retired evidence/coverage/check-results schemas with
	// common result/conclusions validation. Digest: native_checks.go + auditcontract/codec.go at cutover.
	nativePolicyV2SHA256 = "sha256:8694fa9204632b1ad79072d685d20c6e14cbcfcda21179ce013f14d0a728b17e"

	// The output normalizer is version 3 since collection reads stored
	// StageMetrics strictly and every counter as reported: a retained
	// passthrough@1 report without model counters no longer yields derived
	// zeros, and a document that breaks its contract fails collection
	// (EVM-R81). Version 2 made a usage counter whose exact sum exceeds the
	// int64 range unavailable instead of a subtotal (EVM-R240) and was
	// sha256:ccf4bff0f907c5326376fe1ca3923079ffdabb40c35d81a33a1b9650831cf964;
	// version 1 was
	// sha256:276d5fdb3b9da5c99ac30e1468bf28935fa1d287d72be935de64e6bbe9a73a8e.
	// The V3 digest is the SHA-256 of output_collection.go, usage_collection.go
	// and evaldomain/usage.go, concatenated, as of that change.
	normalizerID       = "contractor-output@3"
	normalizerV3SHA256 = "sha256:e980902feccb330a94376ecae497965d1698a1d9b20faf3479eb9b573db01295"
)
