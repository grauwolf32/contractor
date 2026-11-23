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

	// Version 4 adds separate terminal summarizer usage to the normal-loop
	// counters (EVM-R75). Its fixed digest hashes output_collection.go,
	// usage_collection.go and evaldomain/usage.go, concatenated at this change.
	// Version 3 used strict stored counters; its digest was
	// sha256:e980902feccb330a94376ecae497965d1698a1d9b20faf3479eb9b573db01295.
	normalizerID       = "contractor-output@4"
	normalizerV4SHA256 = "sha256:7795f7ddecc4c22f0420da944b3cf7b1a10959849b3a4eb83baaedb37db4baaa"
)
