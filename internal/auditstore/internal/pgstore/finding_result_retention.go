package pgstore

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/blob"
	artifactpolicy "github.com/grauwolf32/contractor/internal/artifacts/policy"
	artifactpostgres "github.com/grauwolf32/contractor/internal/artifacts/postgres"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
)

type retainedFindingTask struct {
	task       auditstore.ExactArtifact
	origin     json.RawMessage
	runID      *string
	provenance json.RawMessage
}

// retainFindingCheckResults runs in the collection transaction, after its
// authority/receipt gate. It saves ordinary immutable output associations;
// no result writes a current Finding field, including incidental discoveries.
func (s *ClaimStore) retainFindingCheckResults(ctx context.Context, params auditstore.CollectParams) error {
	var members []string
	for _, item := range params.Items {
		if len(item.FindingAssociations) != 0 {
			members = append(members, item.ExecutionItemID)
		}
	}
	if len(members) == 0 {
		return nil
	}
	var projectID string
	if err := s.db.QueryRow(ctx, "SELECT project_id FROM audits WHERE audit_id=$1", params.Claim.AuditID).Scan(&projectID); err != nil {
		return err
	}
	if _, err := s.db.Exec(ctx, "SELECT contractor_lock_finding_correlation_fence($1)", projectID); err != nil {
		return err
	}
	service := artifacts.NewService(artifactpostgres.NewRepository(s.db))
	scope, err := artifacts.ProjectScope(projectID)
	if err != nil {
		return err
	}
	project, err := service.Project(projectID)
	if err != nil {
		return err
	}
	expected := make(map[string]auditstore.ExactArtifact)
	var materialRefs []contracts.ArtifactRef
	addMaterial := func(value auditstore.ExactArtifact) error {
		if previous, found := expected[value.Ref.Key()]; found {
			if previous.Verify(value) != nil {
				return blob.ErrIntegrity
			}
			return nil
		}
		expected[value.Ref.Key()] = value
		materialRefs = append(materialRefs, value.Ref)
		return nil
	}
	for _, link := range params.Retained {
		if err := addMaterial(link.Artifact); err != nil {
			return err
		}
	}
	for _, item := range params.Items {
		if item.Result != nil {
			if err := addMaterial(*item.Result); err != nil {
				return err
			}
		}
	}
	materialMetadata, err := project.DescribeExact(ctx, materialRefs)
	if err != nil {
		return err
	}
	for _, value := range materialMetadata {
		if expected[value.Ref.Key()].Verify(value) != nil {
			return blob.ErrIntegrity
		}
	}
	retained := make(map[string]auditstore.ExactArtifact)
	retain := func(source auditstore.ExactArtifact) (auditstore.ExactArtifact, error) {
		if previous, exists := retained[source.Ref.Key()]; exists {
			if source.Verify(contracts.ExactArtifact{Ref: source.Ref, Digest: previous.Digest, MediaType: previous.MediaType, SizeBytes: previous.SizeBytes}) != nil {
				return auditstore.ExactArtifact{}, blob.ErrIntegrity
			}
			return previous, nil
		}
		fork, err := service.Fork(ctx, artifacts.Fork{Kind: artifacts.LineageAuditImport,
			Source: scope, SourceRef: source.Ref, Target: scope,
			TargetRef: contracts.ArtifactRef{Namespace: artifactpolicy.FindingMaterialsNamespace, Name: auditdomain.DeterministicID("material", source.Ref.Key())},
			Frozen:    true, Replay: true})
		if err != nil {
			return auditstore.ExactArtifact{}, err
		}
		if fork.MediaType != source.MediaType || fork.Size != source.SizeBytes {
			return auditstore.ExactArtifact{}, blob.ErrIntegrity
		}
		value := source
		value.Ref = fork.TargetRef
		retained[source.Ref.Key()] = value
		return value, nil
	}

	rows, err := s.db.Query(ctx, findingResultTasksSQL, params.Claim.AuditID, params.ExecutionID, members)
	if err != nil {
		return err
	}
	tasks := make(map[string]retainedFindingTask, len(members))
	var refs []contracts.ArtifactRef
	var taskKeys []string
	for rows.Next() {
		var id string
		var ref []byte
		var value retainedFindingTask
		if err := rows.Scan(&id, &ref, &value.task.Digest, &value.origin, &value.runID, &value.provenance); err != nil {
			rows.Close()
			return err
		}
		if strictjson.Decode(ref, &value.task.Ref) != nil || value.task.Ref.ValidateExact() != nil {
			rows.Close()
			return blob.ErrIntegrity
		}
		tasks[id] = value
		taskKeys = append(taskKeys, id)
		refs = append(refs, value.task.Ref)
	}
	rows.Close()
	if err := rows.Err(); err != nil {
		return err
	}
	if len(tasks) != len(members) {
		return auditstore.ErrPrecondition
	}
	descriptors, err := project.DescribeExact(ctx, refs)
	if err != nil {
		return err
	}
	for index, descriptor := range descriptors {
		value := tasks[taskKeys[index]]
		if descriptor.Digest != value.task.Digest {
			return blob.ErrIntegrity
		}
		value.task, err = retain(descriptor)
		if err != nil {
			return err
		}
		tasks[taskKeys[index]] = value
	}

	evidence := make(map[string][]auditstore.ExactArtifact)
	for _, link := range params.Retained {
		var provenance map[string]json.RawMessage
		if strictjson.Decode(link.SourceProvenance, &provenance) != nil {
			return auditstore.ErrInvalid
		}
		var memberID string
		if raw := provenance["executionItemId"]; raw != nil {
			if strictjson.Decode(raw, &memberID) != nil {
				return auditstore.ErrInvalid
			}
		} else if raw := provenance["source"]; raw != nil {
			var source map[string]json.RawMessage
			if strictjson.Decode(raw, &source) != nil {
				return auditstore.ErrInvalid
			}
			if raw := source["executionItemId"]; raw != nil {
				if strictjson.Decode(raw, &memberID) != nil {
					return auditstore.ErrInvalid
				}
			}
		}
		if _, selected := tasks[memberID]; !selected {
			continue
		}
		exact, err := retain(link.Artifact)
		if err != nil {
			return err
		}
		evidence[memberID] = append(evidence[memberID], exact)
	}
	for _, item := range params.Items {
		if len(item.FindingAssociations) == 0 {
			continue
		}
		if item.Result == nil {
			return auditstore.ErrInvalid
		}
		result, err := retain(*item.Result)
		if err != nil {
			return err
		}
		task := tasks[item.ExecutionItemID]
		for _, association := range item.FindingAssociations {
			source := struct {
				AuditID         string          `json:"auditId"`
				ExecutionID     string          `json:"executionId"`
				ExecutionItemID string          `json:"executionItemId"`
				RunID           *string         `json:"runId"`
				Provenance      json.RawMessage `json:"runProvenance"`
				Origin          json.RawMessage `json:"itemOrigin"`
				Semantic        string          `json:"semanticAssessment"`
			}{params.Claim.AuditID, params.ExecutionID, item.ExecutionItemID, task.runID, task.provenance, task.origin, association.SemanticAssessment}
			body, err := json.Marshal(struct {
				Schema    string         `json:"schema"`
				ReceiptID string         `json:"receiptId"`
				Kind      string         `json:"kind"`
				Original  any            `json:"original"`
				Materials map[string]any `json:"materials"`
			}{auditdomain.FindingSourceHistorySchema, association.ReceiptID, "check-result", source,
				map[string]any{"result": result, "task": task.task, "evidence": evidence[item.ExecutionItemID]}})
			if err != nil {
				return err
			}
			name := auditdomain.FindingSourceHistoryName(association.ReceiptID, "check-result", association.AssessmentID)
			if _, err := service.CreateFrozen(ctx, scope, contracts.ArtifactRef{Namespace: artifactpolicy.FindingHistoryNamespace, Name: name},
				artifacts.Payload{MediaType: "application/json", Data: body}); err != nil {
				return fmt.Errorf("retain ordinary finding check result: %w", err)
			}
		}
	}
	_, err = s.db.Exec(ctx, "SELECT contractor_advance_finding_correlation(ARRAY[$1::text])", projectID)
	return err
}
