package artifacts

// CanReplace reports whether the supplied metadata admits a generic exact-CAS
// replacement in this scope. It performs no reads and is informational: Write
// still validates current state, namespace admission and the CAS precondition.
// The caller owns any additional endpoint or lifecycle restrictions.
func (s ScopedStore) CanReplace(metadata Metadata) bool {
	return metadata.Current && !metadata.Frozen &&
		!genericWriteReserved(s.scope, metadata.Ref.Namespace, true)
}
