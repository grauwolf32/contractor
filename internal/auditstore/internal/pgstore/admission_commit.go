package pgstore

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/strictjson"
)

func (s *AdmissionSession) commit(ctx context.Context, mutation auditstore.AdmissionMutation) (auditstore.AdmissionProgress, error) {
	old := s.State
	if len(mutation.Decisions) > auditstore.MaxAdmissionPrimary || len(mutation.Retained) > auditstore.MaxAdmissionRelated {
		return old, auditstore.ErrInvalid
	}
	next := mutation.Next
	if next.Schema == "" {
		next = old
	}
	if mutation.Restart {
		if !old.Exhausted || old.SelectedCount != 0 || old.Root != nil {
			return old, auditstore.ErrPrecondition
		}
		next = old
		next.Cursor = auditstore.AdmissionCursor{}
		next.SourceCutoffAt = old.SourceCutoffAt
		if err := s.tx.QueryRow(ctx, `SELECT clock_timestamp()`).Scan(&next.SourceCutoffAt); err != nil {
			return old, err
		}
		next.Generation = auditdomain.DeterministicID("admission", old.AuditID, old.PreviousRoundID, next.SourceCutoffAt.String())
		if err := s.tx.QueryRow(ctx, admissionSourceUpperSQL, old.AuditID, next.SourceCutoffAt).Scan(&next.SourceUpper.CreatedAt, &next.SourceUpper.ReceiptID); err != nil {
			return old, err
		}
		next.SourceUpper.Ordinal = 512
		next.Budget = json.RawMessage(`{}`)
		next.Exhausted = false
		next.BlockedCode = ""
		next.BlockedReceiptID = ""
		next.StopCode = ""
	} else if next.Schema != old.Schema || next.Generation != old.Generation || next.AuditID != old.AuditID || next.PreviousRoundID != old.PreviousRoundID || next.RoundID != old.RoundID || next.RoundOrdinal != old.RoundOrdinal ||
		next.ProfileDigest != old.ProfileDigest || next.BaselineDigest != old.BaselineDigest || !next.DeadlineAt.Equal(old.DeadlineAt) || !next.SourceCutoffAt.Equal(old.SourceCutoffAt) || next.SourceUpper != old.SourceUpper {
		return old, auditstore.ErrInvalid
	}
	if !mutation.Restart && next.Cursor != old.Cursor && !s.reader.cursors[next.Cursor] {
		return old, auditstore.ErrInvalid
	}
	// Published selection and database staging are independently writer-derived.
	next.SelectedCount, next.PublishedPages, next.PagesDigest = old.SelectedCount, old.PublishedPages, old.PagesDigest
	next.StagedCount, next.StagedPages, next.Root = old.StagedCount, old.StagedPages, old.Root
	phases := 0
	for _, present := range []bool{mutation.Page != nil, mutation.Root != nil, mutation.StagePageOrdinal != nil} {
		if present {
			phases++
		}
	}
	if phases > 1 || mutation.Page == nil && len(mutation.PageContent) != 0 || mutation.Restart && phases != 0 {
		return old, auditstore.ErrInvalid
	}
	if mutation.Page != nil {
		if old.Root != nil || old.Exhausted {
			return old, auditstore.ErrPrecondition
		}
		page, err := auditdomain.DecodePreparedRoundPage(mutation.PageContent)
		if err != nil {
			return old, err
		}
		if page.AuditID != old.AuditID || page.RoundID != old.RoundID || page.Generation != old.Generation {
			return old, auditstore.ErrInvalid
		}
		if page.PageOrdinal != old.PublishedPages || page.FirstItemOrdinal != old.SelectedCount || page.PreviousPageDigest != old.PagesDigest {
			return old, auditstore.ErrConflict
		}
		if mutation.Page.MediaType != auditdomain.PreparedRoundPageMediaType || contracts.ExactContent(mutation.Page.Ref, mutation.Page.MediaType, mutation.PageContent).Verify(*mutation.Page) != nil {
			return old, auditstore.ErrInvalid
		}
		if err := s.validatePageSources(ctx, page, len(mutation.PageContent)); err != nil {
			return old, err
		}
		next.SelectedCount += len(page.Items)
		next.PublishedPages++
		next.PagesDigest = mutation.Page.Digest
		if err := s.retain(ctx, admissionPageKey(old.Generation, "page", page.PageOrdinal), *mutation.Page, map[string]any{"schema": auditstore.AdmissionProvenanceSchema, "generation": old.Generation, "kind": "page", "ordinal": fmt.Sprintf("%06d", page.PageOrdinal)}); err != nil {
			return old, err
		}
	}
	if mutation.Root != nil {
		if old.Root != nil || old.SelectedCount < 1 || !old.Exhausted {
			return old, auditstore.ErrPrecondition
		}
		if mutation.Root.Validate() != nil || !mutation.Root.HasMetadata() || mutation.Root.MediaType != auditdomain.PreparedRoundRootMediaType || mutation.Root.Ref.Namespace != auditdomain.ArtifactNamespace(old.AuditID) || mutation.Root.SizeBytes < 1 || mutation.Root.SizeBytes > auditdomain.MaximumPreparedRoundRootBytes {
			return old, auditstore.ErrInvalid
		}
		data, err := s.readMetadata(ctx, *mutation.Root)
		if err != nil {
			return old, err
		}
		root, err := auditdomain.DecodePreparedRoundRoot(data)
		if err != nil {
			return old, err
		}
		if root.AuditID != old.AuditID || root.RoundID != old.RoundID || root.Generation != old.Generation || root.ProfileDigest != old.ProfileDigest || root.BaselineDigest != old.BaselineDigest || root.ItemCount != old.SelectedCount || root.PageCount != old.PublishedPages || root.PagesDigest != old.PagesDigest {
			return old, auditstore.ErrInvalid
		}
		if err := s.retain(ctx, admissionPageKey(old.Generation, "root", 0), *mutation.Root, map[string]any{"schema": auditstore.AdmissionProvenanceSchema, "generation": old.Generation, "kind": "root", "ordinal": "000000"}); err != nil {
			return old, err
		}
		next.Root = mutation.Root
	}
	if mutation.StagePageOrdinal != nil {
		if old.Root == nil || old.StagedPages >= old.PublishedPages {
			return old, auditstore.ErrPrecondition
		}
		if *mutation.StagePageOrdinal != old.StagedPages {
			return old, auditstore.ErrConflict
		}
		_, data, err := s.Reads.ReadPage(ctx, *mutation.StagePageOrdinal)
		if err != nil {
			return old, err
		}
		page, err := auditdomain.DecodePreparedRoundPage(data)
		if err != nil {
			return old, err
		}
		if page.AuditID != old.AuditID || page.RoundID != old.RoundID || page.Generation != old.Generation || page.PageOrdinal != old.StagedPages || page.FirstItemOrdinal != old.StagedCount || len(page.Items) > old.SelectedCount-old.StagedCount {
			return old, auditstore.ErrInvalid
		}
		// ReadPage already charged the exact page bytes once. Stage only the
		// trusted stored content, preserving the one published digest chain.
		if err := s.validatePageSources(ctx, page, 0); err != nil {
			return old, err
		}
		if err := s.stageItems(ctx, page); err != nil {
			return old, err
		}
		next.StagedCount += len(page.Items)
		next.StagedPages++
		if next.StagedPages == old.PublishedPages {
			if next.StagedCount != old.SelectedCount {
				return old, auditstore.ErrInvalid
			}
			tag, err := s.tx.Exec(ctx, admissionSealMembershipSQL, old.AuditID, old.RoundID, old.Generation, old.Root.Digest, old.PublishedPages, old.PagesDigest, old.SelectedCount)
			if err != nil {
				return old, err
			}
			if tag.RowsAffected() != 1 {
				return old, auditstore.ErrConflict
			}
		}
	}
	for _, link := range mutation.Retained {
		if err := s.reader.work.add(0, 1, len(mustRoundResultsJSON(link)), 0); err != nil {
			return old, err
		}
		if err := s.retain(ctx, link.LogicalKey, link.Artifact, link.SourceProvenance); err != nil {
			return old, err
		}
	}
	for _, decision := range mutation.Decisions {
		if err := s.recordDecision(ctx, decision); err != nil {
			return old, err
		}
	}
	oldBytes, _ := json.Marshal(old)
	next.Revision = old.Revision
	next.Changed = false
	nextBytes, _ := json.Marshal(next)
	if bytes.Equal(oldBytes, nextBytes) && phases == 0 && len(mutation.Decisions) == 0 && len(mutation.Retained) == 0 {
		return old, nil
	}
	next.Revision = old.Revision + 1
	encoded, err := encodeAdmissionProgress(next)
	if err != nil {
		return old, err
	}
	if err := s.reader.work.add(0, 0, len(encoded), 0); err != nil {
		return old, err
	}
	tag, err := s.tx.Exec(ctx, `UPDATE audit_rounds SET admission_checkpoint=$3::text::json WHERE audit_id=$1 AND round_id=$2 AND (admission_checkpoint->>'revision')::bigint=$4`, old.AuditID, old.PreviousRoundID, string(encoded), int64(old.Revision))
	if err != nil {
		return old, err
	}
	if tag.RowsAffected() != 1 {
		return old, auditstore.ErrConflict
	}
	next.Changed = true
	return next, nil
}

// validatePageSources charges represented descriptors on each changing phase;
// the source reader accounts its bounded exact metadata and payloads separately.
func (s *AdmissionSession) validatePageSources(ctx context.Context, page auditdomain.PreparedRoundPage, metadataBytes int) error {
	if len(page.Items) > s.reader.work.primary {
		s.reader.work.primary = len(page.Items)
	}
	if err := s.reader.work.add(0, auditdomain.PreparedRoundPageDescriptors(page), metadataBytes, 0); err != nil {
		return err
	}
	for _, item := range page.Items {
		if err := s.validatePreparedSource(ctx, item); err != nil {
			return err
		}
	}
	return nil
}

func (s *AdmissionSession) readMetadata(ctx context.Context, exact auditstore.ExactArtifact) ([]byte, error) {
	if err := s.reader.work.add(0, 1, int(exact.SizeBytes), 0); err != nil {
		return nil, err
	}
	store, _ := s.Artifacts.Project(s.reader.project)
	read, err := store.ReadExact(ctx, exact)
	if err != nil {
		return nil, err
	}
	return read.Payload.Data, nil
}

func (s *AdmissionSession) retain(ctx context.Context, key string, exact auditstore.ExactArtifact, provenance any) error {
	if key == "" || exact.Validate() != nil || !exact.HasMetadata() || exact.Ref.Namespace != auditdomain.ArtifactNamespace(s.State.AuditID) {
		return auditstore.ErrInvalid
	}
	store, _ := s.Artifacts.Project(s.reader.project)
	metadata, err := store.Metadata(ctx, exact.Ref)
	if err != nil {
		return err
	}
	if !metadata.Frozen || !metadata.Current || exact.Verify(metadata.Exact()) != nil {
		return auditstore.ErrInvalid
	}
	var encoded []byte
	if raw, ok := provenance.(json.RawMessage); ok {
		encoded = raw
	} else {
		encoded, err = json.Marshal(provenance)
		if err != nil {
			return err
		}
	}
	var same, found bool
	if err := s.tx.QueryRow(ctx, admissionLinkReplaySQL, s.State.AuditID, key, mustRoundResultsJSON(exact.Ref), exact.Digest, encoded).Scan(&found, &same); err != nil {
		return err
	}
	if found {
		if !same {
			return auditstore.ErrConflict
		}
		return nil
	}
	var charged bool
	if err := s.tx.QueryRow(ctx, admissionExactChargedSQL, s.State.AuditID, exact.Ref.Namespace, exact.Ref.Name, *exact.Ref.Revision, exact.Digest).Scan(&charged); err != nil {
		return err
	}
	if !charged {
		tag, err := s.tx.Exec(ctx, `UPDATE audits SET retained_evidence_bytes=retained_evidence_bytes+$2 WHERE audit_id=$1 AND retained_evidence_bytes+$2<=max_evidence_bytes`, s.State.AuditID, exact.SizeBytes)
		if err != nil {
			return err
		}
		if tag.RowsAffected() != 1 {
			return auditstore.ErrEvidenceBudgetExhausted
		}
	}
	_, err = s.tx.Exec(ctx, `INSERT INTO audit_artifact_links(audit_id,logical_key,artifact_ref,artifact_digest,media_type,size_bytes,source_provenance) VALUES($1,$2,$3::jsonb,$4,$5,$6,$7::jsonb)`, s.State.AuditID, key, mustRoundResultsJSON(exact.Ref), exact.Digest, exact.MediaType, exact.SizeBytes, encoded)
	return err
}

func (s *AdmissionSession) recordDecision(ctx context.Context, decision auditstore.AdmissionDecision) error {
	if decision.ProposedCheckOrdinal < 0 || decision.ProposedCheckOrdinal >= 512 || decision.Proposal.Validate() != nil || decision.SourceCheckDigest == "" || validateID("receipt", decision.ReceiptID) != nil || validateText("decision code", decision.Code, 128, true) != nil {
		return auditstore.ErrInvalid
	}
	if decision.Disposition != "admitted" && decision.Disposition != "rejected" && decision.Disposition != "deferred" && decision.Disposition != "blocked" {
		return auditstore.ErrInvalid
	}
	projection, ok := s.reader.indexes[decision.ReceiptID]
	if !ok {
		return auditstore.ErrInvalid
	}
	var heldBytes []byte
	if err := s.tx.QueryRow(ctx, `SELECT proposal_ref FROM finding_proposal_audit_holds WHERE audit_id=$1 AND receipt_id=$2`, s.State.AuditID, decision.ReceiptID).Scan(&heldBytes); err != nil {
		return err
	}
	var held auditstore.ExactArtifact
	if strictjson.Decode(heldBytes, &held) != nil {
		return auditstore.ErrInvalid
	}
	if decision.ProposedCheckOrdinal >= len(projection.Checks) || decision.Proposal.Verify(held) != nil || decision.SourceCheckDigest != projection.Checks[decision.ProposedCheckOrdinal].Fragment.Digest || decision.PlanDigest != "" && decision.PlanDigest != projection.Checks[decision.ProposedCheckOrdinal].PlanDigest {
		return auditstore.ErrInvalid
	}
	keySource := admissionSourceKey(decision.ReceiptID, decision.ProposedCheckOrdinal, projection.ProposalDigest, decision.SourceCheckDigest)
	identity := auditdomain.DigestBytes([]byte(keySource + "\n" + admissionPolicyDigest(s.State) + "\n" + decision.Code))
	generation := s.State.Generation
	key := "admission/" + generation + "/decision/" + decision.Disposition + "/" + identity[7:]
	if decision.Disposition == "rejected" {
		generation = ""
		key = "admission/decision/rejected/" + identity[7:]
	}
	value := struct {
		Schema         string                       `json:"schema"`
		AuditID        string                       `json:"audit_id"`
		Generation     string                       `json:"generation"`
		ProfileDigest  string                       `json:"profile_digest"`
		BaselineDigest string                       `json:"baseline_digest"`
		Decision       auditstore.AdmissionDecision `json:"decision"`
	}{auditstore.AdmissionDecisionSchema, s.State.AuditID, generation, s.State.ProfileDigest, s.State.BaselineDigest, decision}
	data, err := strictjson.Canonical(value)
	if err != nil {
		return err
	}
	if err := s.reader.work.add(0, 1, len(data), 0); err != nil {
		return err
	}
	scope, _ := artifacts.ProjectScope(s.reader.project)
	written, err := s.Artifacts.CreateFrozen(ctx, scope, contracts.ArtifactRef{Namespace: auditdomain.ArtifactNamespace(s.State.AuditID), Name: "admission-decision-" + auditdomain.DigestBytes([]byte(key))[7:]}, artifacts.Payload{MediaType: "application/json", Data: data})
	if err != nil {
		return err
	}
	exact := contracts.ExactContent(written.Ref, written.MediaType, data)
	return s.retain(ctx, key, exact, map[string]any{"schema": auditstore.AdmissionDecisionSchema, "generation": generation, "source_key": keySource, "policy_digest": admissionPolicyDigest(s.State), "disposition": decision.Disposition, "receipt_id": decision.ReceiptID, "ordinal": decision.ProposedCheckOrdinal, "code": decision.Code})
}
