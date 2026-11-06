package auditimport

import (
	"context"
	"fmt"

	"github.com/grauwolf32/contractor/internal/artifacts"
	artifactpolicy "github.com/grauwolf32/contractor/internal/artifacts/policy"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
)

// Extend only the trusted per-item evidence membership with exact, frozen
// captures minted by the allocation-bound Artifact publisher. JSON references
// alone confer no authority. Retention uses the ordinary claimed Artifact fork.
func (i *Importer) taskHTTPContext(ctx context.Context, audit auditstore.Audit, execution auditstore.Execution, trusted, raw []byte) ([]byte, map[string]auditcontract.Material, error) {
	invalid := func() ([]byte, map[string]auditcontract.Material, error) {
		return nil, nil, fmt.Errorf("%w: invalid selected HTTP evidence", ErrPermanent)
	}
	if auditcontract.ValidateResultShape("contractor.audit.conclusions.v1", raw) != nil {
		return invalid()
	}
	var contextValue, conclusions map[string]any
	if strictjson.Decode(trusted, &contextValue) != nil || strictjson.Decode(raw, &conclusions) != nil {
		return invalid()
	}
	members := map[string]map[string]any{}
	for _, value := range contextValue["items"].([]any) {
		member := value.(map[string]any)
		members[member["item_key"].(string)] = member
	}
	selected := map[string]auditcontract.Material{}
	retentions := []RunRetention{}
	var total int64
	for _, value := range conclusions["items"].([]any) {
		item := value.(map[string]any)
		member, found := members[item["item_key"].(string)]
		if !found {
			return invalid()
		}
		for _, evidence := range item["evidence"].([]any) {
			data, err := strictjson.Canonical(evidence)
			if err != nil {
				return invalid()
			}
			var material auditcontract.Material
			if strictjson.Decode(data, &material) != nil {
				return invalid()
			}
			if material.ScopeKind != "run" || material.Ref.Namespace != artifactpolicy.HTTPExchangeNamespace {
				continue
			}
			if material.ScopeID != *execution.RunID || material.MediaType != auditdomain.HTTPExchangeMediaType {
				return invalid()
			}
			key := string(data)
			if _, exists := selected[key]; !exists {
				total += material.SizeBytes
				if len(selected) >= 256 || total > auditcontract.MaximumDocumentBytes {
					return invalid()
				}
				exact, frozen, err := i.artifacts.DescribeRunBinding(ctx, *execution.RunID, material.Ref)
				if err != nil {
					if artifacts.IsExactReferenceRejection(err) {
						return invalid()
					}
					return nil, nil, err
				}
				if !frozen || contracts.MaterialKey(material) != contracts.MaterialKey(scopedHTTPMaterial("run", *execution.RunID, exact)) {
					return invalid()
				}
				_, payload, err := i.artifacts.ReadRunExact(ctx, *execution.RunID, material.Ref)
				if err != nil {
					if artifacts.IsExactReferenceRejection(err) {
						return invalid()
					}
					return nil, nil, err
				}
				envelope, err := auditdomain.DecodeHTTPExchangeArtifact(payload)
				if err != nil || envelope.Origin.RunID != *execution.RunID || envelope.Capture.Source != "http" || material.Ref.Name != auditdomain.HTTPExchangeArtifactName(envelope.Origin.InvocationID, envelope.Capture.Source, envelope.Capture.RequestID) {
					return invalid()
				}
				name, err := auditdomain.MaterialArtifactName(material)
				if err != nil {
					if artifacts.IsExactReferenceRejection(err) {
						return invalid()
					}
					return nil, nil, err
				}
				selected[key] = material
				retentions = append(retentions, RunRetention{Source: exact, Target: contracts.ArtifactRef{Namespace: auditdomain.ArtifactNamespace(execution.AuditID), Name: name}})
			}
			// Repeated selection by batch members shares one physical retained source.
			evidenceList := member["evidence"].([]any)
			exists := false
			for _, allowed := range evidenceList {
				encoded, _ := strictjson.Canonical(allowed)
				if string(encoded) == key {
					exists = true
					break
				}
			}
			if !exists {
				member["evidence"] = append(evidenceList, evidence)
			}
		}
	}
	retained, err := i.artifacts.RetainRunExact(ctx, *execution.RunID, audit.ProjectID, retentions)
	if err != nil {
		if artifacts.IsExactReferenceRejection(err) {
			return invalid()
		}
		return nil, nil, err
	}
	if len(retained) != len(retentions) {
		return invalid()
	}
	translations := map[string]auditcontract.Material{}
	for index, retention := range retentions {
		original := scopedHTTPMaterial("run", *execution.RunID, retention.Source)
		data, _ := strictjson.Canonical(original)
		translations[string(data)] = scopedHTTPMaterial("audit", execution.AuditID, retained[index])
	}
	data, err := strictjson.Canonical(contextValue)
	return data, translations, err
}

func scopedHTTPMaterial(kind, id string, exact contracts.ExactArtifact) contracts.MaterialDescriptor {
	return contracts.MaterialDescriptor{ScopeKind: kind, ScopeID: id, Ref: exact.Ref, Digest: exact.Digest, MediaType: exact.MediaType, SizeBytes: exact.SizeBytes}
}
