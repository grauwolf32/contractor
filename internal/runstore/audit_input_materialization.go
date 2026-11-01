package runstore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

const (
	AuditInputInitializationPendingReason = "audit_input_initialization_pending"
	AuditInputInitializationFailedReason  = "audit_input_initialization_failed"
	MaxAuditInputMaterializationBytes     = 256 << 10
	MaxAuditInputMaterializationRefs      = 1024
	AuditInputMaterializationTimeout      = 10 * time.Minute
)

// AuditInputMaterialization is durable, bounded progress of an Audit-owned
// initializer. Checkpoint is its strict domain schema, not an artifact reader
// instruction. Only Refs establish Run-local history holds.
type AuditInputMaterialization struct {
	Identity   string                  `json:"identity" strictjson:"required"`
	Revision   uint32                  `json:"revision" strictjson:"required"`
	Complete   bool                    `json:"complete" strictjson:"required"`
	DeadlineAt time.Time               `json:"deadlineAt" strictjson:"required"`
	Checkpoint json.RawMessage         `json:"checkpoint" strictjson:"required"`
	Refs       []contracts.ArtifactRef `json:"refs" strictjson:"required"`
}

func validateAuditInputMaterialization(value AuditInputMaterialization) ([]byte, error) {
	if !contentdigest.Valid(value.Identity) || value.Revision == 0 || value.DeadlineAt.IsZero() ||
		value.Refs == nil || len(value.Refs) > MaxAuditInputMaterializationRefs ||
		len(value.Checkpoint) > MaxAuditInputMaterializationBytes {
		return nil, invalidf("Audit input materialization identity or bounds are invalid")
	}
	var checkpoint map[string]json.RawMessage
	if strictjson.Decode(value.Checkpoint, &checkpoint) != nil || checkpoint == nil || strictjson.HasNULEscape(value.Checkpoint) {
		return nil, invalidf("Audit input materialization checkpoint is invalid")
	}
	seen := make(map[string]bool, len(value.Refs))
	encodedBytes := len(value.Checkpoint) + len(value.Identity) + 128
	for _, ref := range value.Refs {
		if ref.ValidateExact() != nil || seen[ref.Key()] {
			return nil, invalidf("Audit input materialization references are invalid")
		}
		seen[ref.Key()] = true
		encoded, err := json.Marshal(ref)
		if err != nil {
			return nil, invalidf("Audit input materialization reference cannot be encoded")
		}
		encodedBytes += len(encoded) + 1
		if encodedBytes > MaxAuditInputMaterializationBytes {
			return nil, invalidf("Audit input materialization exceeds its byte bound")
		}
	}
	data, err := json.Marshal(value)
	if err != nil || len(data) > MaxAuditInputMaterializationBytes {
		return nil, invalidf("Audit input materialization exceeds its byte bound")
	}
	return data, nil
}

// GetAuditInputMaterialization locks the child before any materialization work.
// The existing claim-bound Audit row lock precedes this lock; cancellation
// either wins first or waits until the bounded fork transaction finishes.
func (s *PostgresStore) GetAuditInputMaterialization(ctx context.Context, runID string) (*AuditInputMaterialization, error) {
	if err := validateOpaque("runID", runID); err != nil {
		return nil, err
	}
	var data []byte
	if err := s.db.QueryRow(ctx, auditInputMaterializationReadSQL, runID).Scan(&data); err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return nil, &StateConflictError{Resource: "Audit input initialization", ID: runID, Expected: string(RunInitializing)}
		}
		return nil, fmt.Errorf("read Audit input materialization: %w", err)
	}
	if data == nil {
		return nil, nil
	}
	if len(data) > MaxAuditInputMaterializationBytes {
		return nil, invalidf("stored Audit input materialization exceeds its byte bound")
	}
	var value AuditInputMaterialization
	if err := strictjson.Decode(data, &value); err != nil {
		return nil, invalidf("stored Audit input materialization is invalid")
	}
	if _, err := validateAuditInputMaterialization(value); err != nil {
		return nil, err
	}
	return &value, nil
}

// SetAuditInputMaterialization advances one existing child by compare-and-set.
// Its strict history trigger acquires every exact local ref in this same
// transaction, and no Scheduler claim is possible until the closure is complete.
func (s *PostgresStore) SetAuditInputMaterialization(ctx context.Context, runID string, previous uint32, value AuditInputMaterialization) error {
	if err := validateOpaque("runID", runID); err != nil {
		return err
	}
	if previous == ^uint32(0) || value.Revision != previous+1 {
		return invalidf("Audit input materialization revision does not advance once")
	}
	data, err := validateAuditInputMaterialization(value)
	if err != nil {
		return err
	}
	tag, err := s.db.Exec(ctx, auditInputMaterializationWriteSQL, runID, int64(previous), string(data),
		value.Identity, value.Complete, value.DeadlineAt, AuditInputInitializationPendingReason, SkillInitializationPendingReason)
	if err != nil {
		if persistencepostgres.Classify(err) == persistencepostgres.ClassInvalid {
			return fmt.Errorf("record Audit input materialization: %w: %w", ErrInvalid, err)
		}
		return fmt.Errorf("record Audit input materialization: %w", err)
	}
	if tag.RowsAffected() != 1 {
		return &StateConflictError{Resource: "Audit input materialization", ID: runID, Expected: "initializing exact revision"}
	}
	return nil
}
