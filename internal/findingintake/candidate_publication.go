package findingintake

import (
	"context"
	"encoding/json"
	"errors"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/jackc/pgx/v5"
)

type CandidatePublication struct {
	RunID, StageID, ClaimID string
	Candidates, Evidence    contracts.ArtifactRef
	selection               *contracts.ArtifactRef
}

type preparedCandidates struct {
	execution intakeExecution
	batch     CandidateBatch
	meta      contracts.ExactArtifact
	request   CandidatePublication
}

// PublishCandidates authenticates and validates the entire common envelope
// before committing individually receipted assertions.
func (s *Service) PublishCandidates(ctx context.Context, request CandidatePublication) (CandidatePublicationSummary, error) {
	prepared, err := s.prepareCandidates(ctx, request)
	if err != nil {
		return CandidatePublicationSummary{}, err
	}
	return s.publishPreparedCandidates(ctx, prepared)
}
func (s *Service) prepareCandidates(ctx context.Context, request CandidatePublication) (preparedCandidates, error) {
	if !contracts.ValidResourceID(request.RunID) || !contracts.ValidResourceID(request.StageID) || !contracts.ValidResourceID(request.ClaimID) || request.Candidates.ValidateExact() != nil || request.Evidence.ValidateExact() != nil || request.Candidates.SameExact(request.Evidence) {
		return preparedCandidates{}, ErrInvalid
	}
	execution := intakeExecution{RunID: request.RunID, StageID: request.StageID, ClaimID: request.ClaimID, Candidate: request.Candidates, Evidence: request.Evidence, Selection: request.selection}
	// Authenticate the operation and exact Stage inputs before payload reads,
	// including replays. Each mutation repeats this fence in its transaction.
	err := persistencepostgres.InTxWithRetry(ctx, s.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		stage, err := runstore.NewPostgresTxStore(tx, s.storage).LockStageOperation(ctx, request.RunID, request.StageID, request.ClaimID)
		if err != nil {
			return err
		}
		if !pinnedCandidateSelection(stage, execution) {
			return ErrAccessDenied
		}
		run, err := runstore.NewPostgresStore(tx, s.storage).GetRun(ctx, request.RunID)
		if err != nil {
			return err
		}
		if run.ProjectID == nil {
			return ErrAccessDenied
		}
		spec, err := resolved.DecodeResolvedStageSnapshot(stage.StageSpecSnapshot)
		if err != nil || spec.Operation == nil || spec.Operation.Ref != contracts.PublishFindingsOperation {
			return ErrAccessDenied
		}
		return nil
	})
	if err != nil {
		return preparedCandidates{}, err
	}
	scope, _ := artifacts.RunScope(request.RunID)
	store, _ := artifacts.NewService(s.storage.Repository(s.pool)).Run(request.RunID)
	batchMeta, err := store.Metadata(ctx, request.Candidates)
	if err != nil {
		return preparedCandidates{}, err
	}
	evidenceMeta, err := store.Metadata(ctx, request.Evidence)
	if err != nil {
		return preparedCandidates{}, err
	}
	if batchMeta.MediaType != CandidateMediaType || batchMeta.Size > MaxCandidateBatchBytes || evidenceMeta.MediaType != CandidateEvidenceMediaType || evidenceMeta.Size > MaxCandidateBatchBytes {
		return preparedCandidates{}, ErrInvalid
	}
	origin, version, err := s.storage.Repository(s.pool).ExecutionOrigin(ctx, scope, request.Candidates)
	if err != nil {
		return preparedCandidates{}, err
	}
	evidenceOrigin, _, err := s.storage.Repository(s.pool).ExecutionOrigin(ctx, scope, request.Evidence)
	if err != nil {
		return preparedCandidates{}, err
	}
	if origin == nil || evidenceOrigin == nil || !sameScanExecution(origin, evidenceOrigin) {
		return preparedCandidates{}, ErrAccessDenied
	}
	raw, err := store.ReadExact(ctx, batchMeta.Exact())
	if err != nil {
		return preparedCandidates{}, err
	}
	batch, err := decodeCandidateBatch(raw.Payload.Data)
	if err != nil {
		return preparedCandidates{}, err
	}
	if !batch.Evidence.Ref.SameExact(evidenceOrigin.Artifact) || batch.Evidence.Digest != evidenceMeta.Digest || batch.Evidence.MediaType != evidenceMeta.MediaType || batch.Evidence.SizeBytes != evidenceMeta.Size {
		return preparedCandidates{}, ErrAccessDenied
	}
	// Validate bounded common evidence before the first mutation. Its records
	// bind every selected candidate by key; there is no raw provider parser here.
	proof, err := store.ReadExact(ctx, evidenceMeta.Exact())
	if err != nil {
		return preparedCandidates{}, err
	}
	var document struct {
		Schema  string            `json:"schema" strictjson:"required"`
		Records []json.RawMessage `json:"records" strictjson:"required"`
		Inputs  []json.RawMessage `json:"inputs,omitempty"`
	}
	if strictjson.Decode(proof.Payload.Data, &document) != nil || document.Schema != "contractor.finding.scanner-evidence.v1" || document.Records == nil || len(document.Records) > MaxCandidateBatch {
		return preparedCandidates{}, ErrInvalid
	}
	records := map[string]bool{}
	for _, record := range document.Records {
		var fields map[string]json.RawMessage
		if strictjson.Decode(record, &fields) != nil {
			return preparedCandidates{}, ErrInvalid
		}
		var key string
		if strictjson.Decode(fields["key"], &key) != nil || !candidateKey.MatchString(key) || records[key] {
			return preparedCandidates{}, ErrInvalid
		}
		records[key] = true
	}
	for _, candidate := range batch.Candidates {
		if !records[candidate.Key] {
			return preparedCandidates{}, ErrInvalid
		}
		// Reject malformed common assertions before any sibling is committed.
		if _, err := canonicalize(candidateSubmission(candidate, []contracts.ArtifactRef{request.Evidence, request.Candidates}, batch.Complete)); err != nil {
			return preparedCandidates{}, err
		}
	}
	execution.Source, execution.SourceVersion = origin, version
	return preparedCandidates{execution: execution, batch: batch, meta: batchMeta.Exact(), request: request}, nil
}

func (s *Service) publishPreparedCandidates(ctx context.Context, prepared preparedCandidates) (CandidatePublicationSummary, error) {
	execution, batch, batchMeta, request := prepared.execution, prepared.batch, prepared.meta, prepared.request
	version := execution.SourceVersion
	result := CandidatePublicationSummary{Sources: []CandidatePublicationSource{}}
	result.Complete = batch.Complete
	for _, count := range batch.Skipped {
		result.Skipped += count
	}
	for ordinal, candidate := range batch.Candidates {
		if err := ctx.Err(); err != nil {
			return result, err
		}
		fingerprint := candidateFingerprint(candidate, batchMeta, ordinal)
		execution.Fingerprint = &fingerprint
		execution.RequestDigest, _ = contentdigest.CanonicalJSON([]string{version, candidate.Key, batchMeta.Digest})
		input := candidateSubmission(candidate, []contracts.ArtifactRef{request.Evidence, request.Candidates}, batch.Complete)
		receipt, replayed, err := s.submit(ctx, execution, input)
		if err != nil {
			return result, err
		}
		result.Sources = append(result.Sources, CandidatePublicationSource{Key: candidate.Key, ReceiptID: receipt.ReceiptID, FindingID: receipt.FindingID, Error: receipt.PublicationError})
		switch {
		case replayed:
			result.Replayed++
		case receipt.PublicationError != "":
			result.Rejected++
		case receipt.FindingID != nil:
			result.Published++
		default:
			return result, errors.New("Finding publication has no classified outcome")
		}
	}
	return result, nil
}

func sameScanExecution(a, b *artifacts.ExecutionOrigin) bool {
	return a.RunID == b.RunID && a.StageExecutionID == b.StageExecutionID && a.AllocationID == b.AllocationID && a.LogicalAgentName == b.LogicalAgentName
}
