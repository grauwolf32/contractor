package pgstore

import (
	"time"
)

func optionalDeadline(value time.Time) *time.Time {
	if value.IsZero() {
		return nil
	}
	return &value
}
