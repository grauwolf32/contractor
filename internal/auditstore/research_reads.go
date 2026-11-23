package auditstore

import (
	"time"

	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
)

// MaterialEntry names an accepted exact material through the ordinary Audit
// artifact reader. Staged bindings and arbitrary embedded references are absent.
type MaterialEntry struct {
	Ordinal   int64                  `json:"-"`
	Key       string                 `json:"key"`
	Material  auditcontract.Material `json:"material"`
	CreatedAt time.Time              `json:"created_at"`
}

type MaterialListParams struct {
	OwnerID, AuditID string
	AfterOrdinal     int64
	Limit            int
	AuditRevision    *uint64
}

type MaterialPage struct {
	Items         []MaterialEntry
	NextOrdinal   *int64
	AuditRevision uint64
}

// TaskAttempt exposes only the ordinary Run association and collection facts.
// RunDeleted distinguishes retained provenance from a live navigable Run.
type TaskAttempt struct {
	Attempt     int                     `json:"attempt"`
	ExecutionID string                  `json:"execution_id"`
	RunID       *string                 `json:"run_id,omitempty"`
	RunDeleted  bool                    `json:"run_deleted"`
	Outcome     *TerminalOutcome        `json:"outcome,omitempty"`
	Disposition *CollectionDisposition  `json:"collection_disposition,omitempty"`
	State       string                  `json:"state"`
	CreatedAt   time.Time               `json:"created_at"`
	CollectedAt *time.Time              `json:"collected_at,omitempty"`
	Result      *auditcontract.Material `json:"result,omitempty"`
	ResultKey   *string                 `json:"result_key,omitempty"`
	Stop        *auditcontract.TaskStop `json:"stop,omitempty"`
}

type TaskAttemptListParams struct {
	OwnerID, AuditID, TaskID string
	AfterAttempt, Limit      int
	AuditRevision            *uint64
}

type TaskAttemptPage struct {
	Items         []TaskAttempt
	NextAttempt   *int
	AuditRevision uint64
}
