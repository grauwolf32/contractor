package artifacts

import "time"

// StorageTimestamps describes registry facts of an exact revision and its current
// binding. Shared blob/version creation can predate this binding and scope.
// BindingUpdatedAt includes freeze/thaw; it does not date a content change.
type StorageTimestamps struct {
	ScopeCreatedAt   time.Time `json:"scopeCreatedAt"`
	BlobCreatedAt    time.Time `json:"blobCreatedAt"`
	VersionCreatedAt time.Time `json:"versionCreatedAt"`
	BindingUpdatedAt time.Time `json:"bindingUpdatedAt"`
}

// normalizeMetadataDates rejects corrupt stored facts instead of inventing a
// date. PostgreSQL timestamptz can also contain infinity or a year outside the
// public RFC 3339 range; pgx rejects infinity while this checks finite dates.
func normalizeMetadataDates(metadata *Metadata) error {
	for _, date := range []*time.Time{&metadata.BindingCreatedAt, &metadata.CreatedAt,
		&metadata.StorageTimestamps.ScopeCreatedAt, &metadata.StorageTimestamps.BlobCreatedAt,
		&metadata.StorageTimestamps.VersionCreatedAt, &metadata.StorageTimestamps.BindingUpdatedAt} {
		if date.IsZero() || date.Year() < 1 || date.Year() > 9999 {
			return ErrArtifactIntegrity
		}
		*date = date.UTC()
	}
	return nil
}
