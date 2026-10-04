package evalservice

// These pins match the policies shipped before explicit versions were added.
// Change a pin only when its behavior changes intentionally; source edits,
// comments, and refactors must not invalidate existing Experiments.
const (
	humanPolicyV1SHA256  = "sha256:25740d3d9f24f4894b4f44d4e90585df006712bc84017be40114c6c1fea6d8c1"
	nativePolicyV1SHA256 = "sha256:d63f0a408b377c728a9d472819216fd3f34d666bd251321ee3ccadeb66db766f"
	normalizerV1SHA256   = "sha256:276d5fdb3b9da5c99ac30e1468bf28935fa1d287d72be935de64e6bbe9a73a8e"
)
