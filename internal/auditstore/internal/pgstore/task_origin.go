package pgstore

import (
	"context"
	"errors"
	"fmt"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	"github.com/jackc/pgx/v5"
)

// ExecutionAuditOrigin resolves Run-level provenance without selecting a Task.
func (s *QueryStore) ExecutionAuditOrigin(ctx context.Context, executionID, runID string) (string, error) {
	if !contracts.ValidResourceID(executionID) || !contracts.ValidResourceID(runID) {
		return "", auditstore.ErrInvalid
	}
	var auditID string
	err := s.db.QueryRow(ctx, executionAuditOriginSQL, executionID, runID).Scan(&auditID)
	if errors.Is(err, pgx.ErrNoRows) {
		return "", auditstore.ErrNotFound
	}
	return auditID, err
}

// ExecutionTaskOrigin resolves a tool-selected member through immutable Run
// membership. A single-member Run permits omission; a batch requires its key.
// The targeted unique-index lookup reads one task, independent of Audit size.
func (s *QueryStore) ExecutionTaskOrigin(ctx context.Context, executionID, runID, itemKey string) (auditcontract.Task, error) {
	if !contracts.ValidResourceID(executionID) || !contracts.ValidResourceID(runID) {
		return auditcontract.Task{}, auditstore.ErrInvalid
	}
	ordinal := 0
	if itemKey != "" {
		var valid bool
		ordinal, valid = auditcontract.ExecutionItemOrdinal(itemKey)
		if !valid {
			return auditcontract.Task{}, auditstore.ErrInvalid
		}
	}
	var data []byte
	err := s.db.QueryRow(ctx, executionTaskOriginSQL, executionID, runID, ordinal, itemKey == "").Scan(&data)
	if errors.Is(err, pgx.ErrNoRows) {
		return auditcontract.Task{}, auditstore.ErrNotFound
	}
	if err != nil {
		return auditcontract.Task{}, err
	}
	task, err := auditcontract.DecodeTask(data)
	if err != nil {
		return auditcontract.Task{}, fmt.Errorf("%w: invalid selected Task", auditstore.ErrInvalid)
	}
	return task, nil
}
