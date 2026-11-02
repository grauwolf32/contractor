package auditdomain

import (
	"context"
	"strings"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
)

// PreparedOpenAPIInput contains exact ordinary material, not Audit authority.
type PreparedOpenAPIInput struct {
	Source       auditcontract.Material
	Data         []byte
	Settings     *auditcontract.Material
	SettingsData []byte
	Options      auditcontract.OpenAPIWorkArguments
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
		if input.Settings != nil || len(input.SettingsData) != 0 {
			return PreparedOpenAPIWork{}, invalid(CodeInventoryInvalid, "prepared.settings")
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
		if input.Settings == nil || !validPreparedSource(*input.Settings, input.SettingsData) || input.Settings.ScopeID != input.Source.ScopeID || input.Settings.MediaType != JSONMediaType {
			return PreparedOpenAPIWork{}, invalid(CodeInventoryInvalid, "prepared.settings")
		}
		basis, subjects, err = openAPIScanSubjects(input.Data, input.Source.MediaType, input.SettingsData, ExactInput{Name: "settings", Ref: input.Settings.Ref, Digest: input.Settings.Digest}, InventoryOptions{SourceInputName: "openapi", SourceRef: input.Source.Ref, ApprovalRequirement: ApprovalNone}, input.Options.MaxItems)
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
		material := struct {
			Schema    string                  `json:"schema"`
			Source    auditcontract.Material  `json:"source"`
			Settings  *auditcontract.Material `json:"settings,omitempty"`
			Operation *OperationTask          `json:"operation,omitempty"`
			Scan      *ScanTask               `json:"scan,omitempty"`
		}{"contractor.audit.openapi-work-item.v1", input.Source, input.Settings, subject.operation, subject.scan}
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
