package auditpriority

// External package tests retain access to the pure helpers without publishing
// production APIs before a real caller exists.
var (
	ResolveTopN     = resolveTopN
	SelectionDigest = selectionDigest
)
