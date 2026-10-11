package artifacts

// ErrBlobRuntimeMissing lets external tests identify the internal error that a
// filesystem payload operation returns without an installed blob runtime.
var ErrBlobRuntimeMissing = errBlobRuntimeMissing
