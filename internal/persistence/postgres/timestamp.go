package postgres

import "time"

// Timestamp returns value in UTC at PostgreSQL's microsecond precision, so a
// stored and re-read time compares equal to the value that was written.
func Timestamp(value time.Time) time.Time { return value.UTC().Truncate(time.Microsecond) }
