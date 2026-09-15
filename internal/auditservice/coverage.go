package auditservice

import (
	"context"
	"encoding/json"
	"errors"
	"strings"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
)

func (s *Service) describeCoverage(ctx context.Context, projectID string, rows []auditstore.CoverageRow) ([]auditstore.CoverageRow, error) {
	scope, err := artifacts.ProjectScope(projectID)
	if err != nil {
		return nil, err
	}
	repository := artifacts.NewPostgresRepository(s.pool)
	// Deduplicate shared batch results and decode each result archive once. Reads
	// retain the repository's combined byte limit and exact-revision checks.
	tasks, results := make(map[string][]int), make(map[string][]int)
	refs := make(map[string]auditstore.ExactArtifact)
	var keys []string
	for index, row := range rows {
		for kind, ref := range []*auditstore.ExactArtifact{&row.Task, row.Result} {
			if ref == nil {
				continue
			}
			encoded, err := json.Marshal(ref.Ref)
			if err != nil {
				return nil, err
			}
			key := string(encoded) + ":" + ref.Digest
			if _, exists := refs[key]; !exists {
				keys = append(keys, key)
				refs[key] = *ref
			}
			if kind == 0 {
				tasks[key] = append(tasks[key], index)
			} else {
				results[key] = append(results[key], index)
			}
		}
		rows[index].Details = &auditstore.CoverageDetails{Methods: []string{}, Evidence: []auditstore.CoverageEvidence{}}
	}
	var readBatch func([]string) error
	readBatch = func(batch []string) error {
		requests := make([]artifacts.ExactReadRequest, len(batch))
		for index, key := range batch {
			requests[index] = artifacts.ExactReadRequest{Scope: scope, Ref: refs[key].Ref}
		}
		reads, err := repository.ReadExactBatch(ctx, requests)
		if errors.Is(err, artifacts.ErrPayloadTooLarge) && len(batch) > 1 {
			middle := len(batch) / 2
			if err := readBatch(batch[:middle]); err != nil {
				return err
			}
			return readBatch(batch[middle:])
		}
		if err != nil {
			return err
		}
		for index, read := range reads {
			key := batch[index]
			if read.Payload.MediaType != auditdomain.PackageMediaType || digestBytes(read.Payload.Data) != refs[key].Digest {
				return artifacts.ErrArtifactIntegrity
			}
			if len(tasks[key]) != 0 {
				for _, rowIndex := range tasks[key] {
					if err := describeCoverageTask(read.Payload.Data, &rows[rowIndex]); err != nil {
						return err
					}
				}
			}
			if len(results[key]) != 0 {
				pkg, err := auditdomain.DecodeCheckResultPackage(read.Payload.Data)
				if err != nil {
					return artifacts.ErrArtifactIntegrity
				}
				for _, rowIndex := range results[key] {
					if err := describeCoverageResult(pkg, &rows[rowIndex]); err != nil {
						return err
					}
				}
			}
		}
		return nil
	}
	for start := 0; start < len(keys); start += artifacts.MaxExactReadBatchSize {
		if err := readBatch(keys[start:min(start+artifacts.MaxExactReadBatchSize, len(keys))]); err != nil {
			return nil, err
		}
	}
	return rows, nil
}

func describeCoverageTask(payload []byte, row *auditstore.CoverageRow) error {
	pkg, err := auditdomain.ValidatePackage(payload)
	if err != nil || pkg.Manifest.Kind != auditdomain.PackageKindTask || len(pkg.Members()) != 1 {
		return artifacts.ErrArtifactIntegrity
	}
	member, exists := pkg.MemberByID("task-document")
	if !exists || member.Metadata().Path != "task.json" || member.Metadata().MediaType != auditdomain.JSONMediaType {
		return artifacts.ErrArtifactIntegrity
	}
	task, err := auditdomain.DecodeItemTask(member.Data())
	if err != nil || task.ItemKey != row.ItemKey || task.SubjectKey != row.SubjectKey {
		return artifacts.ErrArtifactIntegrity
	}
	details := row.Details
	details.TaskDocument = member.Data()
	switch {
	case task.Finding != nil:
		details.Objective = task.Finding.Objective
		details.Methods = []string{task.Finding.Method}
	case task.Checklist != nil:
		details.Objective = task.Checklist.Statement
		details.Methods = append([]string{}, task.Checklist.AllowedMethods...)
	case task.Operation != nil:
		details.Objective = strings.ToUpper(task.Operation.Method) + " " + task.Operation.Path
		for _, field := range []string{"summary", "description"} {
			if value, ok := task.Operation.Resolved[field].(string); ok && value != "" {
				details.Objective += "\n\n" + value
				break
			}
		}
	default:
		details.Objective = task.Scope["objective"]
	}
	return nil
}

func describeCoverageResult(pkg auditdomain.CheckResultPackage, row *auditstore.CoverageRow) error {
	for _, result := range pkg.Results.Results {
		if result.ItemKey != row.ItemKey || result.SubjectKey != row.SubjectKey {
			continue
		}
		row.Details.ResultSummary = result.Summary
		ids := make(map[string]bool, len(result.EvidenceIDs))
		for _, id := range result.EvidenceIDs {
			ids[id] = true
		}
		for _, evidence := range pkg.Evidence.Evidence {
			if ids[evidence.ID] {
				row.Details.Evidence = append(row.Details.Evidence, auditstore.CoverageEvidence{
					ID: evidence.ID, Kind: evidence.Kind, Summary: evidence.Summary,
				})
			}
		}
		return nil
	}
	return artifacts.ErrArtifactIntegrity
}
