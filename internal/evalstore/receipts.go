package evalstore

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"

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

// Receipts retain the current operation-specific shape. Reading never infers
// typed fields from an old receipt or rewrites the stored response.
func decodeReceipt[T any](r Receipt) (T, error) {
	var result T
	decoder := json.NewDecoder(bytes.NewReader(r.Response))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&result); err != nil {
		return result, err
	}
	var trailing any
	if err := decoder.Decode(&trailing); !errors.Is(err, io.EOF) {
		return result, errReceipt
	}
	return result, nil
}

var errReceipt = errors.New("invalid stored evaluation receipt")

func (r Receipt) Dataset() (DatasetReceipt, error) {
	value, err := decodeReceipt[DatasetReceipt](r)
	if err == nil && (value.DatasetID == "" || value.DatasetRevision == "") {
		err = errReceipt
	}
	if err != nil {
		return DatasetReceipt{}, err
	}
	return value, nil
}

func (r Receipt) Experiment() (ExperimentReceipt, error) {
	value, err := decodeReceipt[ExperimentReceipt](r)
	if err == nil && (value.ExperimentID == "" || value.Revision < 1) {
		err = errReceipt
	}
	if err != nil {
		return ExperimentReceipt{}, err
	}
	return value, nil
}

func (r Receipt) Command() (AcceptedCommandReceipt, error) {
	value, err := decodeReceipt[AcceptedCommandReceipt](r)
	if err == nil && (value.CommandID == "" || value.ExperimentRevision < 1) {
		err = errReceipt
	}
	if err != nil {
		return AcceptedCommandReceipt{}, err
	}
	return value, nil
}

func (r Receipt) Submission() (AcceptedSubmissionReceipt, error) {
	value, err := decodeReceipt[AcceptedSubmissionReceipt](r)
	if err == nil && (value.SubmissionKey == "" || value.ExperimentRevision < 1) {
		err = errReceipt
	}
	if err != nil {
		return AcceptedSubmissionReceipt{}, err
	}
	return value, nil
}
