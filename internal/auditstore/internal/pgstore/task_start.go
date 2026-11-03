package pgstore

import (
	"context"
	"encoding/json"
	"errors"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	postgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

// Activate pins the common baseline through the existing owner Start receipt.
// Initial Tasks are admitted by the controller's ordinary bounded mutation.
func (s *OwnerStore) Activate(ctx context.Context, p auditstore.ActivateParams) (auditstore.Audit, bool, error) {
	if validateText("ownerID", p.OwnerID, 256, true) != nil || !contracts.ValidResourceID(p.AuditID) || p.ExpectedRevision == 0 || p.ExpectedRevision > contracts.MaxExactInteger || validateIdempotency(p.IdempotencyKey, p.RequestDigest) != nil || len(p.InitialRetained) > auditstore.MaxArtifactLinksPerCall || validateJSONObject("Audit baseline", p.BaselineSnapshot, auditstore.MaxSnapshotBytes) != nil {
		return auditstore.Audit{}, false, auditstore.ErrInvalid
	}
	bytes, err := validateArtifactLinks(p.InitialRetained)
	if err != nil {
		return auditstore.Audit{}, false, err
	}
	return s.keyedAuditMutation(ctx, p.OwnerID, auditstore.MutationStart, p.IdempotencyKey, p.RequestDigest, func(tx pgx.Tx) (auditMutationResult, error) {
		audit, err := NewQueryStore(tx, s.storage).Get(ctx, p.OwnerID, p.AuditID)
		if err != nil {
			return auditMutationResult{}, err
		}
		pinned, err := decodeAuditProgram(audit)
		if err != nil {
			return auditMutationResult{}, err
		}
		var fields map[string]json.RawMessage
		var schema string
		var inputs map[string]contracts.ExactArtifact
		if strictjson.Decode(p.BaselineSnapshot, &fields) != nil || strictjson.Decode(fields["schema"], &schema) != nil || schema != "contractor.audit.baseline.v1" || strictjson.Decode(fields["inputs"], &inputs) != nil || inputs == nil || fields["inventory"] != nil {
			return auditMutationResult{}, auditstore.ErrInvalid
		}
		for name, slot := range pinned.Program.Inputs {
			input, found := inputs[name]
			if slot.Required && !found || found && (validateExactArtifact("baseline input", input, true) != nil || !contracts.AcceptsMediaType(slot.MediaTypes, input.MediaType)) {
				return auditMutationResult{}, auditstore.ErrInvalid
			}
		}
		for name := range inputs {
			if _, found := pinned.Program.Inputs[name]; !found {
				return auditMutationResult{}, auditstore.ErrInvalid
			}
		}
		if pinned.Program.SeedSource != nil && fields["initial_assertions"] == nil {
			return auditMutationResult{}, auditstore.ErrInvalid
		}
		if _, err := auditdomain.ExecutionProgram(pinned.Program, p.BaselineSnapshot); err != nil {
			return auditMutationResult{}, auditstore.ErrInvalid
		}
		if raw := fields["initial_assertions"]; raw != nil {
			var plan auditcontract.InitialAssertions
			if strictjson.Decode(raw, &plan) != nil || plan.Source.ScopeID != p.AuditID {
				return auditMutationResult{}, auditstore.ErrInvalid
			}
			scope, _ := artifacts.AuditScope(p.AuditID)
			metadata, err := s.storage.TxRepository(tx).Metadata(ctx, scope, plan.Source.Ref)
			if err != nil {
				return auditMutationResult{}, err
			}
			want := contracts.ExactArtifact{Ref: plan.Source.Ref, Digest: plan.Source.Digest, MediaType: plan.Source.MediaType, SizeBytes: plan.Source.SizeBytes}
			if !metadata.Frozen || want.Verify(metadata.Exact()) != nil {
				return auditMutationResult{}, auditstore.ErrInvalid
			}
		}
		links, _ := prepareRetainedLinks(auditstore.CollectParams{Retained: p.InitialRetained})
		started, err := scanAudit(postgres.SavepointQueryRow(ctx, tx, activateAuditSQL, p.OwnerID, p.AuditID, p.ExpectedRevision, []byte(p.BaselineSnapshot), optionalDeadline(p.DeadlineAt), links, bytes))
		if postgres.Classify(err) == postgres.ClassDeleting {
			err = auditstore.ErrProjectDeleting
		}
		if errors.Is(err, pgx.ErrNoRows) || postgres.Classify(err) == postgres.ClassDuplicate {
			err = auditstore.ErrPrecondition
		}
		if err != nil {
			return auditMutationResult{}, writeError("activate Audit", err)
		}
		return auditMutationResult{audit: started, resourceID: started.AuditID, response: map[string]string{"auditId": started.AuditID}}, nil
	})
}
