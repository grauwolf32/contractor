package settingsstore

import (
	"errors"
	"math"
	"testing"
	"time"
)

func TestScanSchedulerSettingsValidatesClosedShape(t *testing.T) {
	t.Parallel()
	now := time.Date(2026, time.September, 5, 12, 0, 0, 0, time.FixedZone("test", 3*60*60))
	settings, err := scanSchedulerSettings(fakeRow{values: []any{2, "7", now}})
	if err != nil {
		t.Fatal(err)
	}
	if settings.MaxConcurrentRuns != 2 || settings.Revision != 7 ||
		settings.UpdatedAt.Location() != time.UTC || !settings.UpdatedAt.Equal(now) {
		t.Fatalf("settings = %+v", settings)
	}

	for _, test := range []struct {
		name   string
		values []any
	}{
		{name: "zero concurrency", values: []any{0, "1", now}},
		{name: "large concurrency", values: []any{33, "1", now}},
		{name: "zero revision", values: []any{1, "0", now}},
		{name: "overflow revision", values: []any{1, "18446744073709551616", now}},
		{name: "invalid revision", values: []any{1, "1.5", now}},
		{name: "zero time", values: []any{1, "1", time.Time{}}},
	} {
		t.Run(test.name, func(t *testing.T) {
			if _, err := scanSchedulerSettings(fakeRow{values: test.values}); !errors.Is(err, ErrInvariant) {
				t.Fatalf("error = %v", err)
			}
		})
	}
}

func TestUpdateSchedulerSettingsRejectsInvalidRequestBeforeStorage(t *testing.T) {
	t.Parallel()
	store := &PostgresStore{}
	for _, params := range []UpdateSchedulerSettingsParams{
		{MaxConcurrentRuns: 0, ExpectedRevision: 1},
		{MaxConcurrentRuns: 33, ExpectedRevision: 1},
		{MaxConcurrentRuns: 1, ExpectedRevision: 0},
		{MaxConcurrentRuns: 1, ExpectedRevision: math.MaxUint64},
	} {
		if _, err := store.UpdateSchedulerSettings(t.Context(), params); !errors.Is(err, ErrInvalid) {
			t.Fatalf("params %+v: error = %v", params, err)
		}
	}
}

type fakeRow struct {
	values []any
	err    error
}

func (r fakeRow) Scan(targets ...any) error {
	if r.err != nil {
		return r.err
	}
	if len(r.values) != 3 || len(targets) != 3 {
		return errors.New("unexpected scan shape")
	}
	*(targets[0].(*int)) = r.values[0].(int)
	*(targets[1].(*string)) = r.values[1].(string)
	*(targets[2].(*time.Time)) = r.values[2].(time.Time)
	return nil
}
