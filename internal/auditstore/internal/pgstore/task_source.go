package pgstore

import (
	"bytes"
	"context"
	"errors"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	postgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

// RecordPreparedSource pins one exact declared work output on the existing
// execution row. It records no Task and permits terminal collection during
// cancellation; AcceptTasks separately enforces the intake fence.
func (s *OwnerStore) RecordPreparedSource(ctx context.Context, p auditstore.PreparedSourceParams) (auditstore.PreparedSourceCursor, error) {
	exact := contracts.ExactArtifact{Ref: p.Source.Ref, Digest: p.Source.Digest, MediaType: p.Source.MediaType, SizeBytes: p.Source.SizeBytes}
	if validateClaimIdentity(p.Claim) != nil || !contracts.ValidResourceID(p.ExecutionID) || p.Count < 0 || p.Count > auditcontract.MaxPreparedWork || p.Source.ScopeKind != "run" || !contracts.ValidResourceID(p.Source.ScopeID) || p.Source.MediaType != "application/json" || p.Source.SizeBytes > auditcontract.MaxPreparedWorkBytes || !exact.HasMetadata() || exact.Validate() != nil {
		return auditstore.PreparedSourceCursor{}, auditstore.ErrInvalid
	}
	source, err := strictjson.Canonical(p.Source)
	if err != nil || len(source) > auditstore.MaxArtifactRefBytes {
		return auditstore.PreparedSourceCursor{}, auditstore.ErrInvalid
	}
	var result auditstore.PreparedSourceCursor
	err = postgres.Within(ctx, s.db, pgx.TxOptions{}, func(tx pgx.Tx) error {
		if err := auditClaims.Lock(ctx, tx, p.Claim.AuditID, p.Claim.HolderID, int64(p.Claim.Epoch)); err != nil {
			if errors.Is(err, postgres.ErrControllerClaimLost) {
				return auditstore.ErrClaimLost
			}
			return err
		}
		audit, err := scanAudit(tx.QueryRow(ctx, `SELECT `+prefixedAuditColumns("audit")+` FROM audits AS audit WHERE audit_id=$1 FOR UPDATE`, p.Claim.AuditID))
		if err != nil {
			return err
		}
		pinned, err := decodeAuditProgram(audit)
		if err != nil {
			return err
		}
		var prior []byte
		var count *int
		execution, err := ScanExecution(withColumns{row: tx.QueryRow(ctx, `SELECT `+ExecutionColumns("execution")+`,prepared_work_source,prepared_work_count,prepared_work_offset FROM audit_executions AS execution WHERE audit_id=$1 AND execution_id=$2 FOR UPDATE`, audit.AuditID, p.ExecutionID), after: []any{&prior, &count, &result.Offset}})
		if errors.Is(err, pgx.ErrNoRows) {
			return auditstore.ErrNotFound
		}
		if err != nil {
			return err
		}
		if execution.Role != auditstore.ExecutionTask || execution.State != auditstore.ExecutionCollecting && execution.State != auditstore.ExecutionCollected || execution.TerminalOutcome == nil || *execution.TerminalOutcome != auditstore.TerminalSucceeded || execution.RunID == nil || *execution.RunID != p.Source.ScopeID || p.Source.Ref.Namespace != "outputs" || p.Source.Ref.Name != pinned.Program.Workflows[execution.WorkflowRole].Outputs["work"] {
			return auditstore.ErrPrecondition
		}
		result.Count = p.Count
		result.Accepting = execution.State == auditstore.ExecutionCollecting && (audit.State == auditstore.AuditActive || audit.State == auditstore.AuditPaused)
		if prior != nil {
			var original auditcontract.Material
			if strictjson.Decode(prior, &original) != nil || count == nil {
				return errStoredTask
			}
			canonical, err := strictjson.Canonical(original)
			if err != nil || !bytes.Equal(canonical, source) || *count != p.Count {
				return auditstore.ErrConflict
			}
		} else {
			if execution.State != auditstore.ExecutionCollecting {
				return auditstore.ErrPrecondition
			}
			if _, err := tx.Exec(ctx, recordPreparedSourceSQL, audit.AuditID, p.ExecutionID, source, p.Count); err != nil {
				return err
			}
		}
		if live, err := auditClaims.Live(ctx, tx, audit.AuditID, p.Claim.HolderID, int64(p.Claim.Epoch)); err != nil {
			return err
		} else if !live {
			return auditstore.ErrClaimLost
		}
		return nil
	})
	return result, err
}
