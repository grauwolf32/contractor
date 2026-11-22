package auditimport

import (
	"context"
	"fmt"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
)

// Extend membership only from Server-accepted, exact check associations. Neither
// a source read nor an arbitrary embedded Audit reference grants this authority.
func (i *Importer) taskSelectedContext(ctx context.Context, execution auditstore.Execution, trusted []byte) ([]byte, map[string]auditcontract.Material, error) {
	links, err := i.reads.ListExecutionEvidence(ctx, execution.AuditID, execution.ExecutionID)
	if err != nil {
		return nil, nil, err
	}
	var value map[string]any
	if strictjson.Decode(trusted, &value) != nil {
		return nil, nil, fmt.Errorf("%w: invalid context", ErrPermanent)
	}
	members := map[string]map[string]any{}
	for _, raw := range value["items"].([]any) {
		member := raw.(map[string]any)
		members[member["task_id"].(string)] = member
	}
	translations := map[string]auditcontract.Material{}
	var total int64
	for _, link := range links {
		total += link.Artifact.SizeBytes
		if total > auditcontract.MaximumDocumentBytes {
			return nil, nil, fmt.Errorf("%w: evidence limit", ErrPermanent)
		}
		data, err := i.artifacts.ReadAuditExact(ctx, execution.AuditID, link.Artifact)
		if err != nil {
			return nil, nil, err
		}
		var record auditcontract.EvidenceRecord
		if strictjson.Decode(data, &record) != nil || record.Validate() != nil || record.Schema != "contractor.audit.evidence.v1" || record.Origin.AuditID != execution.AuditID || record.Origin.ExecutionID != execution.ExecutionID || execution.RunID == nil || record.Origin.RunID != *execution.RunID {
			return nil, nil, fmt.Errorf("%w: invalid evidence origin", ErrPermanent)
		}
		member, ok := members[record.Origin.TaskID]
		if !ok {
			return nil, nil, fmt.Errorf("%w: invalid evidence member", ErrPermanent)
		}
		material := auditcontract.Material{ScopeKind: "audit", ScopeID: execution.AuditID, Ref: link.Artifact.Ref, Digest: link.Artifact.Digest, MediaType: link.Artifact.MediaType, SizeBytes: link.Artifact.SizeBytes}
		for _, selected := range append(record.Sources, material) {
			if selected.ScopeKind != "audit" || selected.ScopeID != execution.AuditID || selected.Ref.ValidateExact() != nil {
				return nil, nil, fmt.Errorf("%w: invalid evidence scope", ErrPermanent)
			}
			encoded, _ := strictjson.Canonical(selected)
			if _, ok := translations[string(encoded)]; !ok {
				translations[string(encoded)] = selected
			}
			list := member["evidence"].([]any)
			found := false
			for _, prior := range list {
				raw, _ := strictjson.Canonical(prior)
				if string(raw) == string(encoded) {
					found = true
					break
				}
			}
			if !found {
				member["evidence"] = append(list, selected)
			}
		}
		// The trusted origin includes the exact assigned hypothesis, if any.
		hypothesis, exists := member["hypothesis"]
		if exists != (record.Origin.Hypothesis != nil) {
			return nil, nil, fmt.Errorf("%w: evidence hypothesis", ErrPermanent)
		}
		if exists {
			full := hypothesis.(map[string]any)
			raw, _ := strictjson.Canonical(map[string]any{"id": full["id"], "revision": full["revision"]})
			var pinned auditcontract.HypothesisRef
			if strictjson.Decode(raw, &pinned) != nil || pinned != *record.Origin.Hypothesis {
				return nil, nil, fmt.Errorf("%w: evidence hypothesis", ErrPermanent)
			}
		}
	}
	data, err := strictjson.Canonical(value)
	return data, translations, err
}
