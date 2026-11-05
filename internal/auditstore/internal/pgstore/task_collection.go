package pgstore

import (
	"encoding/json"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
)

func ValidateCollect(p auditstore.CollectParams) error {
	if len(p.Tasks) == 0 && p.Disposition != auditstore.CollectionContractInvalid {
		return auditstore.ErrInvalid
	}
	if validateClaimIdentity(p.Claim) != nil || !contracts.ValidResourceID(p.ReceiptID) || !contracts.ValidResourceID(p.ExecutionID) || !p.Disposition.Valid() || !contentdigest.Valid(p.RequestDigest) || len(p.Tasks) > auditstore.MaxCollectionItems || len(p.Retained) > auditstore.MaxArtifactLinksPerCall {
		return auditstore.ErrInvalid
	}
	if (p.Disposition == auditstore.CollectionAccepted || p.Disposition == auditstore.CollectionInvalidResult) != (p.SourceOutput != nil) {
		return auditstore.ErrInvalid
	}
	if p.SourceOutput != nil && validateExactArtifact("source output", *p.SourceOutput, true) != nil {
		return auditstore.ErrInvalid
	}
	if p.ErrorCode != nil && validateText("error code", *p.ErrorCode, 128, true) != nil {
		return auditstore.ErrInvalid
	}
	for _, link := range p.Retained {
		if link.Scope != artifacts.ScopeAudit {
			return auditstore.ErrInvalid
		}
	}
	if _, err := validateArtifactLinks(p.Retained); err != nil {
		return err
	}
	keys, ids := map[string]bool{}, map[string]bool{}
	for _, task := range p.Tasks {
		if !contracts.ValidResourceID(task.ExecutionItemID) || !contracts.ValidResourceID(task.TaskID) || keys[task.ExecutionItemID] || ids[task.TaskID] {
			return auditstore.ErrInvalid
		}
		keys[task.ExecutionItemID], ids[task.TaskID] = true, true
		if task.Retryable && p.Disposition != auditstore.CollectionExecutionFailed {
			return auditstore.ErrInvalid
		}
		if task.GapCode != nil && validateText("gap code", *task.GapCode, 128, true) != nil {
			return auditstore.ErrInvalid
		}
		if task.Result == nil && task.GapCode == nil {
			return auditstore.ErrInvalid
		}
		if task.Result != nil {
			material := task.Result
			exact := contracts.ExactArtifact{Ref: material.Ref, Digest: material.Digest, MediaType: material.MediaType, SizeBytes: material.SizeBytes}
			if material.ScopeKind != "audit" || material.ScopeID != p.Claim.AuditID || !exact.HasMetadata() || exact.Validate() != nil {
				return auditstore.ErrInvalid
			}
			retained := false
			for _, link := range p.Retained {
				retained = retained || link.Scope == artifacts.ScopeAudit && link.Artifact.Ref.SameExact(exact.Ref) && link.Artifact.Digest == exact.Digest && link.Artifact.MediaType == exact.MediaType && link.Artifact.SizeBytes == exact.SizeBytes
			}
			if !retained {
				return auditstore.ErrInvalid
			}
		}
	}
	encoded, err := strictjson.Canonical(struct {
		Tasks    []auditstore.TaskCollection
		Retained []auditstore.ArtifactLink
	}{p.Tasks, p.Retained})
	if err != nil || len(encoded) > auditcontract.MaxMutationBytes {
		return auditstore.ErrInvalid
	}
	return nil
}

func prepareTaskCollection(p auditstore.CollectParams) json.RawMessage {
	items := make([]map[string]any, 0, len(p.Tasks))
	for _, task := range p.Tasks {
		item := map[string]any{"execution_item_id": task.ExecutionItemID, "task_id": task.TaskID, "disposition": p.Disposition, "retryable": task.Retryable, "gap_code": task.GapCode}
		if task.Result != nil {
			item["task_result"] = task.Result
			if p.Disposition == auditstore.CollectionAccepted {
				item["result_ref"] = task.Result.Ref
				item["result_digest"] = task.Result.Digest
			}
		}
		items = append(items, item)
	}
	encoded, _ := json.Marshal(items)
	return encoded
}
