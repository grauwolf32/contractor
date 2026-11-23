package pgstore

import (
	"context"
	"encoding/json"
	"errors"
	"sort"

	"github.com/grauwolf32/contractor/internal/artifacts"
	artifactpostgres "github.com/grauwolf32/contractor/internal/artifacts/postgres"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	postgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/jackc/pgx/v5"
)

// LookupTaskAcceptance reconciles a lost acknowledgement before the service
// reads mutable sources. Its receipt belongs to the existing Audit family.
func (s *OwnerStore) LookupTaskAcceptance(ctx context.Context, ownerID, auditID, key, digest string) (auditstore.TaskAcceptance, bool, error) {
	if validateIdempotency(key, digest) != nil || validateID("auditID", auditID) != nil || ownerID == "" {
		return auditstore.TaskAcceptance{}, false, auditstore.ErrInvalid
	}
	receipt, found, err := postgres.FindReceipt(ctx, s.db, auditReceiptKey(ownerID, auditstore.MutationAcceptTasks, key), digest)
	if err != nil || !found {
		return auditstore.TaskAcceptance{}, found, auditReceiptError(err, auditstore.MutationAcceptTasks)
	}
	if receipt.AuditID != auditID {
		return auditstore.TaskAcceptance{}, false, contracts.ErrIdempotencyConflict
	}
	result, err := decodeTaskAcceptance(receipt.Response)
	return result, true, err
}

func decodeTaskAcceptance(data []byte) (auditstore.TaskAcceptance, error) {
	var result auditstore.TaskAcceptance
	if strictjson.Decode(data, &result) != nil || result.TaskIDs == nil || len(result.TaskIDs) > auditcontract.MaxMutationRecords || result.CreatedCount < 0 || result.CreatedCount > len(result.TaskIDs) || result.AuditRevision == 0 || result.AuditRevision > contracts.MaxExactInteger {
		return auditstore.TaskAcceptance{}, errStoredTask
	}
	seen := map[string]bool{}
	for _, id := range result.TaskIDs {
		if !contracts.ValidResourceID(id) || seen[id] {
			return auditstore.TaskAcceptance{}, errStoredTask
		}
		seen[id] = true
	}
	return result, nil
}

// AcceptTasks is the one mutation used by explicit work and accepted ordinary
// producer outputs. Material retention precedes this bounded transaction.
func (s *OwnerStore) AcceptTasks(ctx context.Context, p auditstore.AcceptTasksParams) (auditstore.TaskAcceptance, bool, error) {
	if len(p.HypothesisWrites) > 0 {
		if p.Claim != nil && p.PreparedProgress == nil || len(p.HypothesisWrites)+len(p.Tasks) > auditcontract.MaxMutationRecords {
			return auditstore.TaskAcceptance{}, false, auditstore.ErrInvalid
		}
		data, err := strictjson.Canonical(struct {
			Tasks      []auditstore.AcceptedTask    `json:"tasks"`
			Hypotheses []auditstore.HypothesisWrite `json:"hypotheses"`
		}{p.Tasks, p.HypothesisWrites})
		if err != nil || len(data) > auditcontract.MaxMutationBytes {
			return auditstore.TaskAcceptance{}, false, auditstore.ErrInvalid
		}
		for _, write := range p.HypothesisWrites {
			referenced := false
			for _, task := range p.Tasks {
				referenced = referenced || task.Hypothesis != nil && task.Hypothesis.ID == write.Document.ID && task.Hypothesis.Revision == write.Document.Revision
			}
			if !referenced && p.PreparedProgress == nil {
				return auditstore.TaskAcceptance{}, false, auditstore.ErrInvalid
			}
			if write.Document.Revision == 1 && (write.Document.Origin.Kind != "manual" && !(p.PreparedProgress != nil && write.Document.Origin.Kind == "run") || write.Fixed) {
				return auditstore.TaskAcceptance{}, false, auditstore.ErrInvalid
			}
		}
	}
	selected, err := validateAcceptedTasks(p)
	if err != nil {
		return auditstore.TaskAcceptance{}, false, err
	}
	var result auditstore.TaskAcceptance
	var replayed bool
	err = postgres.Within(ctx, s.db, pgx.TxOptions{}, func(tx pgx.Tx) error {
		receipt, replay, err := postgres.KeyedMutation(ctx, tx, auditReceiptKey(p.OwnerID, auditstore.MutationAcceptTasks, p.IdempotencyKey), p.RequestDigest,
			func() (postgres.Receipt, error) {
				accepted, err := acceptTasksInTransaction(ctx, tx, p, selected, s.storage)
				if err != nil {
					return postgres.Receipt{}, err
				}
				response, err := strictjson.Canonical(accepted)
				if err != nil {
					return postgres.Receipt{}, err
				}
				return postgres.Receipt{Operation: string(auditstore.MutationAcceptTasks), ResourceID: p.AuditID, OwnerID: p.OwnerID, AuditID: p.AuditID, Response: response}, nil
			})
		if err != nil {
			return auditReceiptError(err, auditstore.MutationAcceptTasks)
		}
		if receipt.AuditID != p.AuditID {
			return contracts.ErrIdempotencyConflict
		}
		result, err = decodeTaskAcceptance(receipt.Response)
		replayed = replay
		return err
	})
	if err != nil {
		return auditstore.TaskAcceptance{}, false, err
	}
	return result, replayed, nil
}

func acceptTasksInTransaction(ctx context.Context, tx pgx.Tx, p auditstore.AcceptTasksParams, selected []validatedTask, storage *artifactpostgres.Storage) (auditstore.TaskAcceptance, error) {
	var projectID string
	err := tx.QueryRow(ctx, `SELECT project_id FROM audits WHERE audit_id=$1 AND owner_id=$2`, p.AuditID, p.OwnerID).Scan(&projectID)
	if errors.Is(err, pgx.ErrNoRows) {
		return auditstore.TaskAcceptance{}, auditstore.ErrNotFound
	}
	if err != nil {
		return auditstore.TaskAcceptance{}, err
	}
	if _, err := projectstore.NewPostgresStore(tx).LockActiveAuditProject(ctx, p.OwnerID, projectID); err != nil {
		return auditstore.TaskAcceptance{}, err
	}
	if p.Claim != nil {
		if err := auditClaims.Lock(ctx, tx, p.AuditID, p.Claim.HolderID, int64(p.Claim.Epoch)); err != nil {
			if errors.Is(err, postgres.ErrControllerClaimLost) {
				err = auditstore.ErrClaimLost
			}
			return auditstore.TaskAcceptance{}, err
		}
	}
	audit, err := scanAudit(tx.QueryRow(ctx, `SELECT `+prefixedAuditColumns("audit")+` FROM audits AS audit WHERE audit_id=$1 AND owner_id=$2 FOR UPDATE`, p.AuditID, p.OwnerID))
	if err != nil {
		return auditstore.TaskAcceptance{}, err
	}
	if audit.State != auditstore.AuditActive && audit.State != auditstore.AuditPaused || p.Claim == nil && audit.Revision != p.ExpectedRevision || p.ContinuationRevision != 0 && audit.Revision != p.ContinuationRevision {
		return auditstore.TaskAcceptance{}, auditstore.ErrPrecondition
	}
	pinned, err := decodeAuditProgram(audit)
	if err != nil {
		return auditstore.TaskAcceptance{}, err
	}
	for _, task := range selected {
		if err := validateTaskBinding(pinned, task.Document); err != nil {
			return auditstore.TaskAcceptance{}, err
		}
	}

	if err := validateTaskIntakeProgress(ctx, tx, audit, pinned, p, selected); err != nil {
		return auditstore.TaskAcceptance{}, err
	}
	if len(p.HypothesisWrites) > 0 {
		if err := persistHypothesisSources(ctx, tx, audit, p.HypothesisWrites, storage); err != nil {
			return auditstore.TaskAcceptance{}, err
		}
		if err := persistHypotheses(ctx, tx, audit, pinned, p.HypothesisWrites, p.PreparedProgress != nil); err != nil {
			return auditstore.TaskAcceptance{}, err
		}
	}
	for _, task := range selected {
		if task.Hypothesis != nil {
			h, err := scanHypothesis(tx.QueryRow(ctx, exactHypothesisSQL, audit.AuditID, task.Hypothesis.ID, task.Hypothesis.Revision))
			if err != nil {
				return auditstore.TaskAcceptance{}, auditstore.ErrInvalid
			}
			a, _ := strictjson.Canonical(h.Document)
			b, _ := strictjson.Canonical(task.Hypothesis)
			if string(a) != string(b) {
				return auditstore.TaskAcceptance{}, auditstore.ErrInvalid
			}
		}
	}
	existing, err := existingTasksBySource(ctx, tx, p.AuditID, selected)
	if err != nil {
		return auditstore.TaskAcceptance{}, err
	}
	result := auditstore.TaskAcceptance{TaskIDs: make([]string, 0, len(selected)), AuditRevision: audit.Revision}
	created := make([]validatedTask, 0, len(selected))
	count, iteration, iterationCount := audit.TaskCount, audit.TaskIteration, audit.IterationTaskCount
	for _, task := range selected {
		result.TaskIDs = append(result.TaskIDs, task.Document.TaskID)
		if prior, found := existing[task.SourceKey]; found {
			if !sameAcceptedTask(prior, task) {
				return auditstore.TaskAcceptance{}, auditstore.ErrConflict
			}
			continue
		}
		if count >= audit.Limits.MaxItemsTotal {
			return auditstore.TaskAcceptance{}, auditstore.ErrTaskBudgetExhausted
		}
		if iterationCount >= audit.Limits.MaxItemsPerRound {
			iteration++
			iterationCount = 0
		}
		if iteration > audit.Limits.MaxRounds {
			return auditstore.TaskAcceptance{}, auditstore.ErrTaskBudgetExhausted
		}
		count++
		iterationCount++
		task.Ordinal, task.Iteration = count, iteration
		created = append(created, task)
	}
	if err := verifyTaskMaterials(ctx, tx, p.AuditID, selected, storage); err != nil {
		return auditstore.TaskAcceptance{}, err
	}
	if len(created) > 0 {
		payload, err := strictjson.Canonical(created)
		if err != nil || len(payload) > auditcontract.MaxMutationBytes {
			return auditstore.TaskAcceptance{}, auditstore.ErrInvalid
		}
		if _, err := tx.Exec(ctx, insertTasksSQL, p.AuditID, payload); err != nil {
			if postgres.Classify(err) == postgres.ClassDuplicate {
				return auditstore.TaskAcceptance{}, auditstore.ErrConflict
			}
			return auditstore.TaskAcceptance{}, err
		}
	}
	if len(created) > 0 || p.InitialProgress != nil || p.PreparedProgress != nil {
		var nextInitial *int
		if p.InitialProgress != nil {
			nextInitial = &p.InitialProgress.To
		}
		var revision, sequence int64
		if err := tx.QueryRow(ctx, advanceAcceptedTasksSQL, p.AuditID, count, iteration, iterationCount, len(created), nextInitial).Scan(&revision, &sequence); err != nil {
			return auditstore.TaskAcceptance{}, err
		}
		result.AuditRevision = uint64(revision)
		if p.PreparedProgress != nil {
			if _, err := tx.Exec(ctx, `UPDATE audit_executions SET prepared_work_offset=$3,updated_at=clock_timestamp() WHERE audit_id=$1 AND execution_id=$2`, p.AuditID, p.PreparedProgress.ExecutionID, p.PreparedProgress.To); err != nil {
				return auditstore.TaskAcceptance{}, err
			}
		}
	}
	if p.Claim != nil {
		live, err := auditClaims.Live(ctx, tx, p.AuditID, p.Claim.HolderID, int64(p.Claim.Epoch))
		if err != nil {
			return auditstore.TaskAcceptance{}, err
		}
		if !live {
			return auditstore.TaskAcceptance{}, auditstore.ErrClaimLost
		}
	}
	result.CreatedCount = len(created)
	return result, nil
}

func existingTasksBySource(ctx context.Context, tx pgx.Tx, auditID string, selected []validatedTask) (map[string]auditstore.Task, error) {
	keys := make([]string, 0, len(selected))
	for _, task := range selected {
		keys = append(keys, task.SourceKey)
	}
	rows, err := tx.Query(ctx, tasksBySourceSQL, auditID, keys)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	result := map[string]auditstore.Task{}
	for rows.Next() {
		task, err := scanTask(rows)
		if err != nil {
			return nil, err
		}
		result[task.SourceKey] = task
	}
	return result, rows.Err()
}

func verifyTaskMaterials(ctx context.Context, tx pgx.Tx, auditID string, selected []validatedTask, storage *artifactpostgres.Storage) error {
	materials := map[string]auditcontract.Material{}
	for _, task := range selected {
		for _, material := range task.Document.Inputs {
			ref, err := strictjson.Canonical(material.Ref)
			if err != nil {
				return auditstore.ErrInvalid
			}
			key := string(ref)
			if prior, found := materials[key]; found {
				left := contracts.ExactArtifact{Ref: prior.Ref, Digest: prior.Digest, MediaType: prior.MediaType, SizeBytes: prior.SizeBytes}
				right := contracts.ExactArtifact{Ref: material.Ref, Digest: material.Digest, MediaType: material.MediaType, SizeBytes: material.SizeBytes}
				if left.Verify(right) != nil {
					return auditstore.ErrInvalid
				}
			}
			materials[key] = material
		}
	}
	keys := make([]string, 0, len(materials))
	for key := range materials {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	scope, _ := artifacts.AuditScope(auditID)
	repository := storage.TxRepository(tx)
	for start := 0; start < len(keys); start += artifacts.MaxExactReadBatchSize {
		chunk := keys[start:min(start+artifacts.MaxExactReadBatchSize, len(keys))]
		refs := make([]contracts.ArtifactRef, 0, len(chunk))
		for _, key := range chunk {
			refs = append(refs, materials[key].Ref)
		}
		described, err := repository.DescribeExact(ctx, scope, refs)
		if err != nil {
			return err
		}
		for index, found := range described {
			material := materials[chunk[index]]
			wanted := contracts.ExactArtifact{Ref: material.Ref, Digest: material.Digest, MediaType: material.MediaType, SizeBytes: material.SizeBytes}
			if wanted.Verify(found) != nil {
				return auditstore.ErrInvalid
			}
		}
	}
	return nil
}

func validateTaskIntakeProgress(ctx context.Context, tx pgx.Tx, audit auditstore.Audit, pinned auditprofile.PinnedProgram, p auditstore.AcceptTasksParams, selected []validatedTask) error {
	if p.ContinuationRevision != 0 {
		if len(selected) != 1 || pinned.Program.Continuation == nil || selected[0].Source.Kind != "continuation" || selected[0].Source.ThroughOrdinal > audit.TaskCount || selected[0].Document.Workflow != pinned.Program.Continuation.Workflow {
			return auditstore.ErrInvalid
		}
	}
	if progress := p.InitialProgress; progress != nil {
		if progress.From != audit.InitialWorkOffset || progress.To-progress.From != len(selected) || progress.To > len(pinned.Program.InitialWork) || progress.From < 0 {
			return auditstore.ErrPrecondition
		}
		for index, task := range selected {
			initial := pinned.Program.InitialWork[progress.From+index]
			if initial.Hypothesis != "" {
				h := auditdomain.InitialHypothesis(audit.AuditID, initial.Hypothesis, pinned.Program.Hypotheses[initial.Hypothesis])
				if task.Document.Hypothesis == nil || task.Document.Hypothesis.ID != h.ID || task.Document.Hypothesis.Revision != 1 {
					return auditstore.ErrInvalid
				}
			} else if task.Document.Hypothesis != nil {
				return auditstore.ErrInvalid
			}
			if task.Source.Key != initial.Key || task.Document.Workflow != initial.Workflow || task.Document.Objective != initial.Objective {
				return auditstore.ErrInvalid
			}
		}
	}
	if progress := p.PreparedProgress; progress != nil {
		var source json.RawMessage
		var offset, count int
		err := tx.QueryRow(ctx, `SELECT prepared_work_source,prepared_work_offset,prepared_work_count FROM audit_executions WHERE audit_id=$1 AND execution_id=$2 AND state='collecting' AND prepared_work_source IS NOT NULL FOR UPDATE`, p.AuditID, progress.ExecutionID).Scan(&source, &offset, &count)
		if errors.Is(err, pgx.ErrNoRows) {
			return auditstore.ErrPrecondition
		}
		if err != nil {
			return err
		}
		if progress.From != offset || progress.From < 0 || progress.To-progress.From != len(selected) || progress.To > count {
			return auditstore.ErrPrecondition
		}
		var expected auditcontract.Material
		if strictjson.Decode(source, &expected) != nil {
			return errStoredTask
		}

		if len(p.HypothesisWrites) > 0 {
			origin := p.HypothesisWrites[0].Document.Origin
			var member bool
			if err := tx.QueryRow(ctx, `SELECT EXISTS(SELECT 1 FROM audit_execution_items WHERE audit_id=$1 AND execution_id=$2 AND task_id=$3)`, p.AuditID, progress.ExecutionID, origin.TaskID).Scan(&member); err != nil {
				return err
			}
			if !member {
				return auditstore.ErrInvalid
			}
			for _, write := range p.HypothesisWrites {
				h := write.Document
				a, ae := strictjson.Canonical(h.Origin.Source)
				b, be := strictjson.Canonical(expected)
				if h.Origin.Kind != "run" || h.Origin.RunID != expected.ScopeID || h.Origin.TaskID != origin.TaskID || ae != nil || be != nil || string(a) != string(b) || write.Fixed {
					return auditstore.ErrInvalid
				}
			}
		}
		for _, task := range selected {
			a, err := strictjson.Canonical(task.Source.Output)
			b, expectedErr := strictjson.Canonical(expected)
			if err != nil || expectedErr != nil || string(a) != string(b) {
				return auditstore.ErrInvalid
			}
		}
	}
	return nil
}
