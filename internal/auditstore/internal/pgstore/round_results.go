package pgstore

import (
	"context"
	"encoding/json"
	"errors"
	"strconv"

	"github.com/grauwolf32/contractor/internal/artifacts"
	artifactpostgres "github.com/grauwolf32/contractor/internal/artifacts/postgres"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

// AdvanceRoundResults captures at most one metadata page. An Audit revision
// change before source sealing discards only bounded staged rows and starts a
// fresh generation. Staged bytes are never published as a successful root.
func (s *ClaimStore) AdvanceRoundResults(ctx context.Context, p auditstore.RoundResultsStepParams) (auditstore.RoundResultsBuild, error) {
	if validateClaimIdentity(p.Claim) != nil || validateID("Round", p.RoundID) != nil {
		return auditstore.RoundResultsBuild{}, invalidf("Round results identity is invalid")
	}
	var result auditstore.RoundResultsBuild
	err := persistencepostgres.Within(ctx, s.db, pgx.TxOptions{}, func(tx pgx.Tx) error {
		revision, err := lockRoundResults(ctx, tx, p)
		if err != nil {
			return err
		}
		state, err := readRoundResults(ctx, tx, p.Claim.AuditID, p.RoundID)
		if err != nil {
			return err
		}
		if state.Sealed {
			result = state
			return nil
		}
		if state.Generation != revision {
			if state.PageCount > 0 {
				// Cleanup is a separate bounded changing phase; never combine
				// removal of 128 staged pages with another source capture.
				if tag, err := tx.Exec(ctx, `WITH batch AS (SELECT ordinal FROM audit_round_result_pages WHERE round_id=$1 ORDER BY ordinal LIMIT 128)
DELETE FROM audit_round_result_pages AS page USING batch WHERE page.round_id=$1 AND page.ordinal=batch.ordinal`, p.RoundID); err != nil {
					return err
				} else if _, err = tx.Exec(ctx, `UPDATE audit_rounds SET results_page_count=results_page_count-$2 WHERE round_id=$1`, p.RoundID, tag.RowsAffected()); err != nil {
					return err
				}
				result = state
				return nil
			}
			if _, err := tx.Exec(ctx, `UPDATE audit_rounds SET results_generation=$2, results_item_count=0, results_metadata_bytes=0, results_page_count=0 WHERE round_id=$1`, p.RoundID, int64(revision)); err != nil {
				return err
			}
			state.Generation, state.ItemCount, state.MetadataBytes, state.Pages = revision, 0, 0, nil
		}
		if state.PageCount >= auditdomain.MaximumRoundResultsPages {
			return invalidf("Round results exceeds its page ceiling")
		}
		last := state.ItemCount - 1
		items, err := readRoundResultsItems(ctx, tx, p.Claim.AuditID, p.RoundID, last)
		if err != nil {
			return err
		}
		if len(items) == 0 {
			return invalidf("settled Round results are incomplete")
		}
		items, err = readRoundResultsRelated(ctx, tx, p.Claim.AuditID, p.RoundID, last, items)
		if err != nil {
			return err
		}
		page := auditdomain.RoundResultsPage{Schema: auditdomain.RoundResultsPageSchema, AuditID: p.Claim.AuditID,
			RoundID: p.RoundID, SourceGeneration: revision, Ordinal: state.PageCount, Items: items}
		var encoded []byte
		for len(page.Items) > 0 {
			encoded, err = auditdomain.EncodeRoundResultsPage(page)
			if err == nil {
				break
			}
			if len(page.Items) == 1 {
				return invalidf("Round result cannot fit one complete page: %v", err)
			}
			page.Items = page.Items[:len(page.Items)-1]
		}
		if state.ItemCount+len(page.Items) > state.ExpectedItemCount || state.MetadataBytes+int64(len(encoded))+2*auditdomain.MaximumRoundResultsDocumentBytes > auditdomain.MaximumRoundResultsBytes {
			return invalidf("Round results exceed their complete snapshot bound")
		}
		if _, err := tx.Exec(ctx, `INSERT INTO audit_round_result_pages
    (audit_id,round_id,source_generation,ordinal,last_item_ordinal,item_count,content,content_bytes)
VALUES ($1,$2,$3,$4,$5,$6,$7,octet_length($7::bytea))`, p.Claim.AuditID, p.RoundID, int64(revision), page.Ordinal, page.Items[len(page.Items)-1].Ordinal, len(page.Items), encoded); err != nil {
			return writeError("capture Round results page", err)
		}
		count, bytes := state.ItemCount+len(page.Items), state.MetadataBytes+int64(len(encoded))
		if _, err := tx.Exec(ctx, `UPDATE audit_rounds SET results_item_count=$2, results_metadata_bytes=$3, results_sealed=$4, results_page_count=results_page_count+1 WHERE round_id=$1`, p.RoundID, count, bytes, count == state.ExpectedItemCount); err != nil {
			return writeError("seal Round results source", err)
		}
		result, err = readRoundResults(ctx, tx, p.Claim.AuditID, p.RoundID)
		return err
	})
	return result, err
}

func lockRoundResults(ctx context.Context, tx pgx.Tx, p auditstore.RoundResultsStepParams) (uint64, error) {
	if err := auditClaims.Lock(ctx, tx, p.Claim.AuditID, p.Claim.HolderID, int64(p.Claim.Epoch)); err != nil {
		if errors.Is(err, persistencepostgres.ErrControllerClaimLost) {
			return 0, auditstore.ErrClaimLost
		}
		return 0, err
	}
	var revision uint64
	err := tx.QueryRow(ctx, `SELECT revision FROM audits WHERE audit_id=$1 AND state='active'
AND dispatch_state='open' AND current_round_id=$2 AND (deadline_at IS NULL OR deadline_at>clock_timestamp()) FOR UPDATE`, p.Claim.AuditID, p.RoundID).Scan(&revision)
	if errors.Is(err, pgx.ErrNoRows) {
		return 0, auditstore.ErrPrecondition
	}
	if err != nil {
		return 0, err
	}
	var round string
	err = tx.QueryRow(ctx, `SELECT round_id FROM audit_rounds WHERE audit_id=$1 AND round_id=$2 AND state='assessing' FOR UPDATE`, p.Claim.AuditID, p.RoundID).Scan(&round)
	if errors.Is(err, pgx.ErrNoRows) {
		return 0, auditstore.ErrPrecondition
	}
	return revision, err
}

func readRoundResults(ctx context.Context, db persistencepostgres.DBTX, auditID, roundID string) (auditstore.RoundResultsBuild, error) {
	state := auditstore.RoundResultsBuild{AuditID: auditID, RoundID: roundID, Pages: []auditstore.RoundResultsBuildPage{}}
	var root, published []byte
	err := db.QueryRow(ctx, `SELECT COALESCE(results_generation,0),results_sealed,results_item_count,
expected_item_count,results_metadata_bytes,results_manifest,results_page_count,results_published_pages
FROM audit_rounds WHERE audit_id=$1 AND round_id=$2`, auditID, roundID).Scan(&state.Generation, &state.Sealed, &state.ItemCount, &state.ExpectedItemCount, &state.MetadataBytes, &root, &state.PageCount, &published)
	if errors.Is(err, pgx.ErrNoRows) {
		return state, auditstore.ErrNotFound
	}
	if err != nil {
		return state, err
	}
	if len(root) > 0 {
		state.Manifest = &auditstore.ExactArtifact{}
		if strictjson.Decode(root, state.Manifest) != nil || state.Manifest.Validate() != nil || !state.Manifest.HasMetadata() {
			return state, invalidf("stored Round results root is invalid")
		}
	}
	var pages []auditdomain.RoundResultsPageRef
	if strictjson.Decode(published, &pages) != nil || len(pages) > state.PageCount {
		return state, invalidf("stored Round results publication progress is invalid")
	}
	for index, page := range pages {
		if page.Ordinal != index || page.ItemCount < 1 || page.ItemCount > auditdomain.MaximumRoundResultsPageItems || page.Artifact.Validate() != nil || !page.Artifact.HasMetadata() {
			return state, invalidf("stored Round results page publication is invalid")
		}
		artifact := page.Artifact.Clone()
		state.Pages = append(state.Pages, auditstore.RoundResultsBuildPage{Ordinal: index, ItemCount: page.ItemCount, ContentBytes: int(artifact.SizeBytes), Artifact: &artifact})
	}
	// One exact seek supplies the next publishable page's metadata. Capture
	// never rescans all earlier staged page rows on every Controller tick.
	if state.Sealed && len(pages) < state.PageCount {
		var page auditstore.RoundResultsBuildPage
		err = db.QueryRow(ctx, `SELECT ordinal,item_count,content_bytes FROM audit_round_result_pages
WHERE round_id=$1 AND ordinal=$2 AND source_generation=$3`, roundID, len(pages), state.Generation).Scan(&page.Ordinal, &page.ItemCount, &page.ContentBytes)
		if err != nil {
			return state, err
		}
		state.Pages = append(state.Pages, page)
	}
	return state, nil
}

func (s *ClaimStore) GetRoundResultsBuild(ctx context.Context, p auditstore.RoundResultsStepParams) (auditstore.RoundResultsBuild, error) {
	if validateClaimIdentity(p.Claim) != nil || validateID("Round", p.RoundID) != nil {
		return auditstore.RoundResultsBuild{}, invalidf("Round results identity is invalid")
	}
	if live, err := s.claimLive(ctx, p.Claim); err != nil {
		return auditstore.RoundResultsBuild{}, err
	} else if !live {
		return auditstore.RoundResultsBuild{}, auditstore.ErrClaimLost
	}
	return readRoundResults(ctx, s.db, p.Claim.AuditID, p.RoundID)
}

func (s *ClaimStore) ReadRoundResultsPage(ctx context.Context, p auditstore.RoundResultsStepParams, generation uint64, ordinal int) ([]byte, error) {
	if validateClaimIdentity(p.Claim) != nil || generation == 0 || ordinal < 0 || ordinal >= auditdomain.MaximumRoundResultsPages {
		return nil, invalidf("Round results page selection is invalid")
	}
	var content, encoded []byte
	var projectID string
	err := s.db.QueryRow(ctx, `SELECT page.content,page.artifact,audit.project_id FROM audit_round_result_pages AS page
JOIN audit_rounds AS round USING(round_id) JOIN audits AS audit ON audit.audit_id=page.audit_id
JOIN audit_controller_claims AS claim ON claim.audit_id=page.audit_id
WHERE page.audit_id=$1 AND page.round_id=$4 AND page.source_generation=$5 AND page.ordinal=$6
AND round.results_sealed AND `+auditClaims.LiveCondition("claim", "$2", "$3"), p.Claim.AuditID, p.Claim.HolderID, p.Claim.Epoch, p.RoundID, generation, ordinal).Scan(&content, &encoded, &projectID)
	if errors.Is(err, pgx.ErrNoRows) {
		return nil, auditstore.ErrPrecondition
	}
	if err != nil {
		return nil, err
	}
	if content != nil {
		return content, nil
	}
	var descriptor auditstore.ExactArtifact
	if strictjson.Decode(encoded, &descriptor) != nil || descriptor.Validate() != nil || !descriptor.HasMetadata() {
		return nil, invalidf("published Round results page is invalid")
	}
	project, err := artifacts.NewService(s.storage.Repository(s.db)).Project(projectID)
	if err != nil {
		return nil, err
	}
	result, err := project.ReadExact(ctx, descriptor)
	if err != nil {
		return nil, err
	}
	return result.Payload.Data, nil
}

// Publication uses existing retained bindings and natural artifact-link keys;
// no extra receipt or hold service is introduced.
func (s *ClaimStore) PublishRoundResultsPage(ctx context.Context, p auditstore.PublishRoundResultsPageParams) error {
	if p.Artifact.Validate() != nil || !p.Artifact.HasMetadata() || p.Artifact.MediaType != "application/json" {
		return invalidf("Round results page publication is invalid")
	}
	return persistencepostgres.Within(ctx, s.db, pgx.TxOptions{}, func(tx pgx.Tx) error {
		if _, err := lockRoundResults(ctx, tx, p.RoundResultsStepParams); err != nil {
			return err
		}
		var content []byte
		var existing []byte
		err := tx.QueryRow(ctx, `SELECT page.content,page.artifact FROM audit_round_result_pages AS page JOIN audit_rounds AS round USING(round_id)
WHERE page.round_id=$1 AND page.source_generation=$2 AND page.ordinal=$3 AND round.results_sealed FOR UPDATE OF page`, p.RoundID, p.Generation, p.Ordinal).Scan(&content, &existing)
		if errors.Is(err, pgx.ErrNoRows) {
			return auditstore.ErrPrecondition
		}
		if err != nil {
			return err
		}
		encoded, _ := json.Marshal(p.Artifact)
		if len(existing) > 0 {
			var saved auditstore.ExactArtifact
			if strictjson.Decode(existing, &saved) != nil || !saved.Ref.SameExact(p.Artifact.Ref) || saved.Digest != p.Artifact.Digest {
				return auditstore.ErrConflict
			}
			return nil
		}
		if auditdomain.DigestBytes(content) != p.Artifact.Digest || int64(len(content)) != p.Artifact.SizeBytes {
			return invalidf("Round results page publication content differs")
		}
		if err := verifyRoundResultsPublication(ctx, tx, p.RoundResultsStepParams, p.Artifact, s.storage); err != nil {
			return err
		}
		if err := publishRoundResultsLink(ctx, tx, p.RoundResultsStepParams, p.Generation, "round-results/"+p.RoundID+"/page/"+strconv.Itoa(p.Ordinal), p.Artifact); err != nil {
			return err
		}
		if _, err = tx.Exec(ctx, `UPDATE audit_round_result_pages SET artifact=$4::jsonb,content=NULL WHERE round_id=$1 AND source_generation=$2 AND ordinal=$3`, p.RoundID, p.Generation, p.Ordinal, encoded); err != nil {
			return writeError("publish Round results page", err)
		}
		page, err := auditdomain.DecodeRoundResultsPage(content)
		if err != nil {
			return err
		}
		descriptor, _ := json.Marshal(auditdomain.RoundResultsPageRef{Ordinal: p.Ordinal, ItemCount: len(page.Items), Artifact: p.Artifact})
		tag, err := tx.Exec(ctx, `UPDATE audit_rounds SET results_published_pages=results_published_pages||jsonb_build_array($3::jsonb)
WHERE round_id=$1 AND jsonb_array_length(results_published_pages)=$2`, p.RoundID, p.Ordinal, descriptor)
		if err != nil {
			return writeError("record Round results publication progress", err)
		}
		if tag.RowsAffected() != 1 {
			return auditstore.ErrPrecondition
		}
		return nil
	})
}

func (s *ClaimStore) PublishRoundResultsManifest(ctx context.Context, p auditstore.PublishRoundResultsManifestParams) error {
	encoded, err := auditdomain.EncodeRoundResultsManifest(p.Content)
	if err != nil || p.Manifest.Validate() != nil || !p.Manifest.HasMetadata() || p.Manifest.Digest != auditdomain.DigestBytes(encoded) || p.Manifest.SizeBytes != int64(len(encoded)) {
		return invalidf("Round results manifest publication is invalid")
	}
	return persistencepostgres.Within(ctx, s.db, pgx.TxOptions{}, func(tx pgx.Tx) error {
		if _, err := lockRoundResults(ctx, tx, p.RoundResultsStepParams); err != nil {
			return err
		}
		state, err := readRoundResults(ctx, tx, p.Claim.AuditID, p.RoundID)
		if err != nil {
			return err
		}
		if state.Manifest != nil {
			if !state.Manifest.Ref.SameExact(p.Manifest.Ref) || state.Manifest.Digest != p.Manifest.Digest {
				return auditstore.ErrConflict
			}
			return nil
		}
		if !state.Sealed || state.Generation != p.Generation || p.Content.AuditID != p.Claim.AuditID || p.Content.RoundID != p.RoundID || p.Content.SourceGeneration != p.Generation || p.Content.ItemCount != state.ItemCount || p.Content.MetadataBytes != state.MetadataBytes || len(p.Content.Pages) != state.PageCount || len(state.Pages) != state.PageCount {
			return auditstore.ErrPrecondition
		}
		for index, page := range state.Pages {
			if page.Artifact == nil || !page.Artifact.Ref.SameExact(p.Content.Pages[index].Artifact.Ref) || page.Artifact.Digest != p.Content.Pages[index].Artifact.Digest {
				return auditstore.ErrPrecondition
			}
		}
		if state.MetadataBytes+int64(len(encoded))*2 > auditdomain.MaximumRoundResultsBytes {
			return invalidf("Round results exceed total metadata ceiling")
		}
		if err := verifyRoundResultsPublication(ctx, tx, p.RoundResultsStepParams, p.Manifest, s.storage); err != nil {
			return err
		}
		if err := publishRoundResultsLink(ctx, tx, p.RoundResultsStepParams, p.Generation, "round-results/"+p.RoundID, p.Manifest); err != nil {
			return err
		}
		root, _ := json.Marshal(p.Manifest)
		_, err = tx.Exec(ctx, `UPDATE audit_rounds SET results_manifest=$2::jsonb WHERE round_id=$1`, p.RoundID, root)
		if err != nil {
			return writeError("accept Round results manifest", err)
		}
		return nil
	})
}

func publishRoundResultsLink(ctx context.Context, tx pgx.Tx, p auditstore.RoundResultsStepParams, generation uint64, key string, artifact auditstore.ExactArtifact) error {
	var charged bool
	if err := tx.QueryRow(ctx, `SELECT EXISTS(SELECT 1 FROM audit_artifact_links WHERE audit_id=$1 AND artifact_ref=$2::jsonb AND artifact_digest=$3)`, p.Claim.AuditID, mustRoundResultsJSON(artifact.Ref), artifact.Digest).Scan(&charged); err != nil {
		return err
	}
	if !charged {
		tag, err := tx.Exec(ctx, `UPDATE audits SET retained_evidence_bytes=retained_evidence_bytes+$2 WHERE audit_id=$1 AND retained_evidence_bytes+$2<=max_evidence_bytes`, p.Claim.AuditID, artifact.SizeBytes)
		if err != nil {
			return err
		}
		if tag.RowsAffected() != 1 {
			return auditstore.ErrEvidenceBudgetExhausted
		}
	}
	_, err := tx.Exec(ctx, publishRoundResultsLinkSQL, p.Claim.AuditID, key, mustRoundResultsJSON(artifact.Ref), artifact.Digest, artifact.MediaType, artifact.SizeBytes, p.RoundID, int64(generation))
	if err != nil {
		return writeError("retain Round results metadata", err)
	}
	return nil
}

func mustRoundResultsJSON(value any) []byte { data, _ := json.Marshal(value); return data }

func readRoundResultsItems(ctx context.Context, tx pgx.Tx, auditID, roundID string, after int) ([]auditdomain.RoundResultItem, error) {
	rows, err := tx.Query(ctx, roundResultsItemsSQL, auditID, roundID, after)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	items := []auditdomain.RoundResultItem{}
	for rows.Next() {
		var item auditdomain.RoundResultItem
		var state string
		var final *string
		var requested, completed, gaps, resultRef, sourceRef []byte
		var resultDigest, resultMedia, memberID, executionID, receipt, disposition, outcome, runID, sourceDigest *string
		var size *int64
		var attempt *int
		err := rows.Scan(&item.ItemID, &item.ItemKey, &item.Ordinal, &item.Kind, &item.SubjectKey, &state, &final, &item.Coverage.Status, &requested, &completed, &gaps, &item.Coverage.Rationale, &resultRef, &resultDigest, &resultMedia, &size, &memberID, &executionID, &attempt, &receipt, &disposition, &outcome, &runID, &sourceRef, &sourceDigest)
		if err != nil {
			return nil, err
		}
		if state != "settled" || final == nil || item.Ordinal != after+1+len(items) {
			return nil, invalidf("Round results member is unsettled or out of order")
		}
		item.FinalDisposition = *final
		item.Attachments = []auditdomain.RoundResultAttachment{}
		item.Findings = []auditdomain.RoundResultFinding{}
		if strictjson.Decode(requested, &item.Coverage.Requested) != nil || strictjson.Decode(completed, &item.Coverage.Completed) != nil || strictjson.Decode(gaps, &item.Coverage.Gaps) != nil {
			return nil, invalidf("Round result coverage is invalid")
		}
		if resultRef != nil {
			if resultDigest == nil || resultMedia == nil || size == nil {
				return nil, invalidf("accepted Round result lacks retained descriptor")
			}
			item.Result = &contracts.ExactArtifact{Digest: *resultDigest, MediaType: *resultMedia, SizeBytes: *size}
			if strictjson.Decode(resultRef, &item.Result.Ref) != nil {
				return nil, invalidf("Round result exact ref is invalid")
			}
		}
		if memberID != nil {
			if executionID == nil || attempt == nil || receipt == nil || disposition == nil || outcome == nil {
				return nil, invalidf("Round result attempt lacks exact receipt")
			}
			item.SelectedAttempt = &auditdomain.RoundResultAttempt{ExecutionID: *executionID, ExecutionItemID: *memberID, Attempt: *attempt, ReceiptID: *receipt, CollectionDisposition: *disposition, TerminalOutcome: *outcome, RunID: runID}
			if sourceRef != nil {
				if sourceDigest == nil {
					return nil, invalidf("Round result source digest is absent")
				}
				item.SelectedAttempt.SourceResult = &contracts.ExactArtifact{Digest: *sourceDigest}
				if item.Result != nil {
					item.SelectedAttempt.SourceResult.MediaType = item.Result.MediaType
					item.SelectedAttempt.SourceResult.SizeBytes = item.Result.SizeBytes
				}
				if strictjson.Decode(sourceRef, &item.SelectedAttempt.SourceResult.Ref) != nil {
					return nil, invalidf("Round result source is invalid")
				}
			}
		}
		items = append(items, item)
	}
	return items, rows.Err()
}

func readRoundResultsRelated(ctx context.Context, tx pgx.Tx, auditID, _ string, _ int, items []auditdomain.RoundResultItem) ([]auditdomain.RoundResultItem, error) {
	// 32 indexed primary rows include at most an item, attempt and result
	// apiece (96 descriptors). The remaining 416 bounds related work even
	// when a single selected attempt has a large accepted closure.
	remaining := auditdomain.MaximumRoundResultsSliceDescriptors - 3*len(items)
	remainingBytes := int64(auditdomain.MaximumRoundResultsSliceBytes / 2)
	for index := range items {
		item := &items[index]
		if item.SelectedAttempt == nil {
			continue
		}
		rows, err := tx.Query(ctx, roundResultsAttachmentsSQL, auditID, item.SelectedAttempt.ExecutionItemID, remaining, remainingBytes)
		if err != nil {
			return nil, err
		}
		read := 0
		for rows.Next() {
			var attachment auditdomain.RoundResultAttachment
			var ref, originalRef []byte
			var included bool
			var key, digest, media, kind, evidenceID, memberID, displayRef *string
			var size *int64
			var metadataBytes int64
			if err := rows.Scan(&included, &key, &ref, &digest, &media, &size, &metadataBytes, &kind, &evidenceID, &memberID, &displayRef, &originalRef); err != nil {
				rows.Close()
				return nil, err
			}
			if !included {
				rows.Close()
				if index == 0 {
					return nil, invalidf("one complete Round result closure exceeds the slice byte budget")
				}
				return items[:index], nil
			}
			if key == nil || digest == nil || media == nil || size == nil {
				rows.Close()
				return nil, invalidf("Round attachment metadata is incomplete")
			}
			attachment.LogicalKey = *key
			if kind != nil {
				attachment.Kind = *kind
			}
			if evidenceID != nil {
				attachment.EvidenceID = *evidenceID
			}
			if memberID != nil {
				attachment.ContentMemberID = *memberID
			}
			if displayRef != nil {
				attachment.DisplayRef = *displayRef
			}
			attachment.Artifact.Digest = *digest
			attachment.Artifact.MediaType = *media
			attachment.Artifact.SizeBytes = *size
			if len(originalRef) > 0 && string(originalRef) != "null" {
				original := attachment.Artifact.Clone()
				if strictjson.Decode(originalRef, &original.Ref) != nil || original.Validate() != nil {
					rows.Close()
					return nil, invalidf("Round attachment original exact ref is invalid")
				}
				attachment.Original = &original
			}
			remainingBytes -= metadataBytes
			if strictjson.Decode(ref, &attachment.Artifact.Ref) != nil {
				rows.Close()
				return nil, invalidf("Round attachment descriptor is invalid")
			}
			attachment.Classification = "accepted-attachment"
			item.Attachments = append(item.Attachments, attachment)
			read++
		}
		err = rows.Err()
		rows.Close()
		if err != nil {
			return nil, err
		}
		remaining -= read
		if remaining == 0 || remainingBytes < 2048 {
			if index == 0 {
				return nil, invalidf("Round result exceeds related descriptor ceiling")
			}
			return items[:index], nil
		}
		rows, err = tx.Query(ctx, roundResultsFindingsSQL, auditID, item.SelectedAttempt.ExecutionItemID, min(remaining, int(remainingBytes/2048)))
		if err != nil {
			return nil, err
		}
		read = 0
		for rows.Next() {
			var finding auditdomain.RoundResultFinding
			if err := rows.Scan(&finding.FindingID, &finding.ProjectID); err != nil {
				rows.Close()
				return nil, err
			}
			item.Findings = append(item.Findings, finding)
			remainingBytes -= 2048
			read++
		}
		err = rows.Err()
		rows.Close()
		if err != nil {
			return nil, err
		}
		remaining -= read
		if remaining == 0 || remainingBytes < 2048 {
			if index == 0 {
				return nil, invalidf("Round result exceeds related descriptor ceiling")
			}
			return items[:index], nil
		}
	}
	return items, nil
}

func verifyRoundResultsPublication(ctx context.Context, tx pgx.Tx, p auditstore.RoundResultsStepParams, descriptor auditstore.ExactArtifact, storage *artifactpostgres.Storage) error {
	if descriptor.Ref.Namespace != auditdomain.ArtifactNamespace(p.Claim.AuditID) || descriptor.MediaType != "application/json" || descriptor.SizeBytes > auditdomain.MaximumRoundResultsDocumentBytes {
		return invalidf("Round results publication is outside its protected namespace or bounds")
	}
	var projectID string
	if err := tx.QueryRow(ctx, `SELECT project_id FROM audits WHERE audit_id=$1`, p.Claim.AuditID).Scan(&projectID); err != nil {
		return err
	}
	project, err := artifacts.NewService(storage.Repository(tx)).Project(projectID)
	if err != nil {
		return err
	}
	metadata, err := project.Metadata(ctx, descriptor.Ref)
	if err != nil {
		return err
	}
	if !metadata.Frozen || !metadata.Current || descriptor.Verify(metadata.Exact()) != nil {
		return invalidf("Round results publication has no exact protected bytes")
	}
	return nil
}
