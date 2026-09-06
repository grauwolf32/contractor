package performance

import (
	"bytes"
	"encoding/json"
	"errors"
	"math"
	"strings"
)

var errInvalidRecord = errors.New("invalid bounded performance record")

func (s Sample) Validate() error {
	if s.Version != 1 || strings.TrimSpace(s.Generation) == "" || len(s.Generation) > 128 || s.ObservedAt.IsZero() {
		return errInvalidRecord
	}
	type group struct {
		value    *Freshness
		interval uint32
	}
	groups := make([]group, 0, 5)
	if s.Process != nil {
		groups = append(groups, group{&s.Process.Freshness, 15})
	}
	if s.Pool != nil {
		groups = append(groups, group{&s.Pool.Freshness, 15})
	}
	if s.Database != nil {
		groups = append(groups, group{&s.Database.Freshness, 60})
	}
	if s.DatabaseSize != nil {
		groups = append(groups, group{&s.DatabaseSize.Freshness, 300})
	}
	if s.HTTP != nil {
		groups = append(groups, group{&s.HTTP.Freshness, 15})
		if s.HTTP.Surfaces[0].Surface != Public || s.HTTP.Surfaces[1].Surface != Private {
			return errInvalidRecord
		}
		for _, surface := range s.HTTP.Surfaces {
			h := surface.Duration
			if h.Buckets[13] != h.Count {
				return errInvalidRecord
			}
			for i, count := range h.Buckets {
				if i > 0 && count < h.Buckets[i-1] {
					return errInvalidRecord
				}
			}
		}
	}
	for _, group := range groups {
		if group.value.Validate() != nil || group.value.IntervalSeconds != group.interval || group.value.LastAttemptAt.After(s.ObservedAt) {
			return errInvalidRecord
		}
	}
	raw, err := json.Marshal(s)
	if err != nil || len(raw) > MaxRecordBytes {
		return errInvalidRecord
	}
	// Every numeric leaf is a nonnegative measurement/count. UseNumber avoids
	// silently converting an out-of-range integer into an approximate float.
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.UseNumber()
	var value any
	if err := decoder.Decode(&value); err != nil || !validNumbers(value) {
		return errInvalidRecord
	}
	return nil
}

func (f Freshness) Validate() error {
	switch f.Status {
	case OK, Partial, Unavailable:
	default:
		return errInvalidRecord
	}
	if f.Reason != nil {
		switch *f.Reason {
		case UnsupportedPlatform, ReadFailed, SamplingGap, CounterReset, MissingBaseline, PermissionDenied, StatisticsDisabled, DatabaseUnavailable, BudgetExceeded, RecordLimit:
		default:
			return errInvalidRecord
		}
	}
	if f.Status == OK && (f.Reason != nil || f.ObservedAt == nil) {
		return errInvalidRecord
	}
	if f.LastAttemptAt.IsZero() || (f.ObservedAt != nil && (f.ObservedAt.IsZero() || f.ObservedAt.After(f.LastAttemptAt))) {
		return errInvalidRecord
	}
	c := f.Coverage
	if c.StartedAt.IsZero() || c.EndedAt.IsZero() || c.EndedAt.Before(c.StartedAt) || c.EndedAt.After(f.LastAttemptAt) || c.ObservedSamples > c.ExpectedSamples || c.DurationSeconds < 0 || math.IsNaN(c.DurationSeconds) || math.IsInf(c.DurationSeconds, 0) {
		return errInvalidRecord
	}
	if f.Status == OK && c.ObservedSamples < c.ExpectedSamples {
		return errInvalidRecord
	}
	switch f.IntervalSeconds {
	case 15, 60, 300:
	default:
		return errInvalidRecord
	}
	return nil
}

func validNumbers(value any) bool {
	switch value := value.(type) {
	case json.Number:
		if !strings.ContainsAny(string(value), ".eE") {
			number, err := value.Int64()
			return err == nil && number >= 0 && number <= 1<<53-1
		}
		number, err := value.Float64()
		return err == nil && number >= 0 && !math.IsNaN(number) && !math.IsInf(number, 0)
	case map[string]any:
		for _, item := range value {
			if !validNumbers(item) {
				return false
			}
		}
	case []any:
		for _, item := range value {
			if !validNumbers(item) {
				return false
			}
		}
	}
	return true
}
