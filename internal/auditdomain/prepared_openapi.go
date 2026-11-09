package auditdomain

import (
	"context"
	"strings"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	"github.com/grauwolf32/contractor/internal/scanplan"
)

// PreparedOpenAPIInput contains exact ordinary material, not Audit authority.
type PreparedOpenAPIInput struct {
	Source     auditcontract.Material
	Data       []byte
	Values     *auditcontract.Material
	ValuesData []byte
	Options    auditcontract.OpenAPIWorkArguments
}

type PreparedOpenAPIItem struct {
	Key       string
	Objective string
	Data      []byte
}

type PreparedOpenAPIWork struct {
	Items []PreparedOpenAPIItem
	Gaps  []string
}

// PreparedOpenAPIOperation is ordinary output of the deterministic preparation
// Workflow. It carries domain selection and provenance, never an Audit task ID.
type PreparedOpenAPIOperation struct {
	Schema    string                  `json:"schema"`
	Source    auditcontract.Material  `json:"source"`
	Values    *auditcontract.Material `json:"request_values,omitempty"`
	Operation *OperationTask          `json:"operation,omitempty"`
	Scan      *ScanTask               `json:"scan,omitempty"`
}

// PrepareOpenAPIScanOperation reproduces the selected request from exact bytes.
// Run forks have different refs; the retained source descriptors bind their
// bytes, media and sizes while the caller enforces Artifact authority.
func PrepareOpenAPIScanOperation(operation, source, valuesData []byte, sourceMediaType, scanner string, parameters map[string]string) (ScanTask, scanplan.PreparedAuditOperation, error) {
	var work PreparedOpenAPIOperation
	if len(operation) > MaximumDocumentBytes || strictjson.DecodeNonNull(operation, &work) != nil || work.Schema != "contractor.audit.openapi-work-item.v1" || work.Scan == nil || work.Operation != nil ||
		!validPreparedSource(work.Source, source) || work.Source.MediaType != sourceMediaType || !validRequestValues(work.Values, valuesData, work.Source.ScopeID) || validateScanTask(*work.Scan, false) != nil {
		return ScanTask{}, scanplan.PreparedAuditOperation{}, invalid(CodeInventoryInvalid, "scan.inputs")
	}
	selection, err := scanplan.CompileAuditSelection(scanner, parameters, valuesData)
	if err != nil {
		return ScanTask{}, scanplan.PreparedAuditOperation{}, invalid(CodeInventoryInvalid, "scan.selection")
	}
	prepared, err := selection.PrepareOperation(source, sourceMediaType, work.Source.Ref, work.Scan.Operation)
	if err != nil || !sameCanonicalValue(work.Scan, openAPIScanTask(selection, prepared)) {
		return ScanTask{}, scanplan.PreparedAuditOperation{}, invalid(CodeInventoryInvalid, "scan.preparation")
	}
	return *work.Scan, prepared, nil
}

func validRequestValues(values *auditcontract.Material, data []byte, scopeID string) bool {
	if values == nil {
		return len(data) == 0
	}
	return validPreparedSource(*values, data) && values.ScopeID == scopeID && values.MediaType == JSONMediaType
}

// BuildPreparedOpenAPIWork reuses the owning operation/request selection rules.
// It constructs only bounded JSON material: no task ZIP, inventory execution
// manifest, Round or persisted task identity is assembled on this path.
func BuildPreparedOpenAPIWork(ctx context.Context, input PreparedOpenAPIInput) (PreparedOpenAPIWork, error) {
	if err := ctx.Err(); err != nil {
		return PreparedOpenAPIWork{}, err
	}
	if !validPreparedSource(input.Source, input.Data) || input.Options.MaxItems < 1 || input.Options.MaxItems > auditcontract.MaxPreparedWork || !contracts.ValidName(input.Options.Workflow) {
		return PreparedOpenAPIWork{}, invalid(CodeInventoryInvalid, "prepared.inputs")
	}
	var basis inventoryBasis
	var subjects []inventorySubject
	var err error
	switch input.Options.Selection {
	case "operations":
		if input.Values != nil || len(input.ValuesData) != 0 {
			return PreparedOpenAPIWork{}, invalid(CodeInventoryInvalid, "prepared.request_values")
		}
		var source openAPISource
		if normalizedMediaType(input.Source.MediaType) == PackageMediaType {
			source, err = openAPISourceFromPackage(input.Data)
		} else {
			var root map[string]any
			root, err = parseOpenAPIDocument(input.Data, input.Source.MediaType)
			source = openAPISource{raw: input.Data, mediaType: input.Source.MediaType, entryPoint: "source/openapi", documents: map[string]map[string]any{"source/openapi": root}}
		}
		if err == nil {
			basis, subjects, err = openAPISubjects(source, input.Options.MaxItems)
		}
	case "scan-requests":
		if !validRequestValues(input.Values, input.ValuesData, input.Source.ScopeID) {
			return PreparedOpenAPIWork{}, invalid(CodeInventoryInvalid, "prepared.request_values")
		}
		var selection scanplan.AuditSelection
		selection, err = scanplan.CompileAuditSelection(input.Options.Scanner, input.Options.Parameters, input.ValuesData)
		if err == nil {
			basis, subjects, err = openAPIScanSubjects(input.Data, input.Source.MediaType, selection, input.Values, InventoryOptions{SourceInputName: "openapi", SourceRef: input.Source.Ref}, input.Options.MaxItems)
		}
	default:
		return PreparedOpenAPIWork{}, invalid(CodeInventoryInvalid, "prepared.selection")
	}
	if err != nil {
		return PreparedOpenAPIWork{}, err
	}
	result := PreparedOpenAPIWork{Items: make([]PreparedOpenAPIItem, 0, len(subjects)), Gaps: arrayStrings(basis.Gaps)}
	total := 0
	for _, subject := range subjects {
		if err := ctx.Err(); err != nil {
			return PreparedOpenAPIWork{}, err
		}
		material := PreparedOpenAPIOperation{"contractor.audit.openapi-work-item.v1", input.Source, input.Values, subject.operation, subject.scan}
		data, err := canonicalOpenAPIJSON(material)
		if err != nil || len(data) > MaximumDocumentBytes {
			return PreparedOpenAPIWork{}, invalid(CodeLimitExceeded, "prepared.material")
		}
		total += len(data)
		if total > MaximumGeneratedBytes {
			return PreparedOpenAPIWork{}, invalid(CodeLimitExceeded, "prepared.materials")
		}
		objective := "Inspect the selected OpenAPI operation"
		if subject.operation != nil {
			objective = "Inspect " + strings.ToUpper(subject.operation.Method) + " " + subject.operation.Path
		}
		if subject.scan != nil {
			objective = "Inspect the selected " + subject.scan.Scanner + " request " + subject.scan.Operation
		}
		result.Items = append(result.Items, PreparedOpenAPIItem{Key: subject.itemKey, Objective: objective, Data: data})
	}
	return result, nil
}

func validPreparedSource(source auditcontract.Material, data []byte) bool {
	return source.ScopeKind == "run" && contracts.ValidResourceID(source.ScopeID) && source.Ref.ValidateExact() == nil && contentdigest.Valid(source.Digest) && source.Digest == DigestBytes(data) && source.SizeBytes == int64(len(data)) && validMediaType(source.MediaType)
}
