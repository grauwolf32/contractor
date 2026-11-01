package sourcezip

import "fmt"

// Limits bound expanded source content independently of the compressed Artifact
// ceiling. The Server sends the same values to allocation source readers.
type Limits struct {
	MaxExpandedBytes int64 `json:"maxExpandedBytes" strictjson:"required"`
	MaxFileBytes     int64 `json:"maxFileBytes" strictjson:"required"`
	MaxEntries       int   `json:"maxEntries" strictjson:"required"`
}

func DefaultLimits() Limits {
	return Limits{MaxExpandedBytes: MaxArchiveBytes, MaxFileBytes: MaxFileBytes, MaxEntries: MaxEntries}
}

func (l Limits) Validate() error {
	if l.MaxExpandedBytes <= 0 || l.MaxExpandedBytes > 1<<30 ||
		l.MaxFileBytes <= 0 || l.MaxFileBytes > 64<<20 || l.MaxFileBytes > l.MaxExpandedBytes ||
		l.MaxEntries <= 0 || l.MaxEntries > 50000 {
		return fmt.Errorf("source limits require expanded bytes in 1..1 GiB, file bytes in 1..64 MiB and not above expanded bytes, and entries in 1..50000")
	}
	return nil
}
