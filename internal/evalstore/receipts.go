package evalstore

import (
	"context"
	"encoding/json"
	"errors"

	"github.com/grauwolf32/contractor/internal/evaldomain"
)

type DatasetReceipt struct {
	DatasetID       string `json:"datasetId"`
	DatasetRevision string `json:"datasetRevision"`
}
type ExperimentReceipt struct {
	ExperimentID string           `json:"experimentId"`
	Revision     int64            `json:"revision"`
	State        evaldomain.State `json:"state"`
}
type AcceptedCommandReceipt struct {
	CommandID          string `json:"commandId"`
	ExperimentRevision int64  `json:"experimentRevision"`
	State              string `json:"state"`
}
type AcceptedSubmissionReceipt struct {
	SubmissionKey      string `json:"submissionKey"`
	ExperimentRevision int64  `json:"experimentRevision"`
	State              string `json:"state"`
}

func mutate[T any](ctx context.Context, s *Store, scope Scope, resource, operation string, id evaldomain.MutationIdentity, fn func() (T, error)) (Receipt, error) {
	return s.mutateJSON(ctx, scope, resource, operation, id, func() (json.RawMessage, error) {
		result, err := fn()
		if err != nil {
			return nil, err
		}
		return json.Marshal(result)
	})
}

// Old receipts are immutable and must still replay after an upgrade. This is
// the only place where the legacy overloaded wire fields are interpreted.
type legacyReceipt struct {
	ID       string `json:"id"`
	Revision int64  `json:"revision"`
	State    string `json:"state"`
}

func decodeReceipt[T any](r Receipt, convert func(legacyReceipt) T) (T, error) {
	var result T
	var legacy legacyReceipt
	if err := json.Unmarshal(r.Response, &legacy); err != nil {
		return result, err
	}
	if legacy.ID != "" {
		return convert(legacy), nil
	}
	err := json.Unmarshal(r.Response, &result)
	return result, err
}

var errReceipt = errors.New("invalid stored evaluation receipt")

func (r Receipt) Dataset() (DatasetReceipt, error) {
	value, err := decodeReceipt(r, func(old legacyReceipt) DatasetReceipt { return DatasetReceipt{old.ID, old.State} })
	if err == nil && (value.DatasetID == "" || value.DatasetRevision == "") {
		err = errReceipt
	}
	return value, err
}
func (r Receipt) Experiment() (ExperimentReceipt, error) {
	value, err := decodeReceipt(r, func(old legacyReceipt) ExperimentReceipt {
		return ExperimentReceipt{old.ID, old.Revision, evaldomain.State(old.State)}
	})
	if err == nil && (value.ExperimentID == "" || value.Revision < 1) {
		err = errReceipt
	}
	return value, err
}
func (r Receipt) Command() (AcceptedCommandReceipt, error) {
	value, err := decodeReceipt(r, func(old legacyReceipt) AcceptedCommandReceipt {
		return AcceptedCommandReceipt{old.ID, old.Revision, old.State}
	})
	if err == nil && (value.CommandID == "" || value.ExperimentRevision < 1) {
		err = errReceipt
	}
	return value, err
}
func (r Receipt) Submission() (AcceptedSubmissionReceipt, error) {
	value, err := decodeReceipt(r, func(old legacyReceipt) AcceptedSubmissionReceipt {
		return AcceptedSubmissionReceipt{old.ID, old.Revision, old.State}
	})
	if err == nil && (value.SubmissionKey == "" || value.ExperimentRevision < 1) {
		err = errReceipt
	}
	return value, err
}
