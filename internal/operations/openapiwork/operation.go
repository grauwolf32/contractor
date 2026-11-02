// Package openapiwork binds ordinary Stage artifacts to the OpenAPI domain's
// deterministic preparation. It owns neither Audit admission nor execution.
package openapiwork

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/grauwolf32/contractor/internal/apierror"
	"github.com/grauwolf32/contractor/internal/artifactpolicy"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	"github.com/grauwolf32/contractor/internal/operations"
	"github.com/grauwolf32/contractor/internal/stageresult"
	"github.com/jackc/pgx/v5/pgxpool"
)

type Module struct{ pool *pgxpool.Pool }

func New(pool *pgxpool.Pool) *Module { return &Module{pool: pool} }
func (*Module) Ref() string          { return contracts.OpenAPIPrepareWorkOperation }

func invalid() error {
	return &operations.Failure{Code: "invalid_request", Message: "OpenAPI preparation inputs, arguments or outputs are invalid"}
}

func (m *Module) Run(ctx context.Context, call operations.Call) (stageresult.StageContentResult, error) {
	result, err := m.prepare(ctx, call)
	if err == nil {
		return result, nil
	}
	var domain *auditdomain.ValidationError
	if errors.As(err, &domain) || errors.Is(err, contracts.ErrValidation) {
		return stageresult.StageContentResult{}, invalid()
	}
	if kind, ok := apierror.Of(err); ok {
		k := kind.ErrorKind()
		return stageresult.StageContentResult{}, &operations.Failure{Code: k.Code(), Message: k.Message(), Retryable: k.Retryable()}
	}
	return stageresult.StageContentResult{}, err
}

func (m *Module) prepare(ctx context.Context, call operations.Call) (stageresult.StageContentResult, error) {
	empty := stageresult.StageContentResult{}
	options, err := auditcontract.OpenAPIWorkOptions(call.Arguments)
	if err != nil || len(call.Outputs) != 1 {
		return empty, invalid()
	}
	output, exists := call.Outputs["work"]
	if !exists || output.Validate() != nil || output.Revision != nil || artifactpolicy.IsPurposeReservedNamespace(output.Namespace) {
		return empty, invalid()
	}
	inputs := 1
	if options.Selection == "scan-requests" {
		inputs = 2
	}
	if len(call.Inputs) != inputs || call.Inputs["openapi"].ValidateExact() != nil || inputs == 2 && call.Inputs["settings"].ValidateExact() != nil {
		return empty, invalid()
	}
	if m.pool == nil {
		return empty, &operations.Failure{Code: "server_operation_unavailable", Message: "OpenAPI preparation storage is unavailable"}
	}
	input, err := m.readInputs(ctx, call, options)
	if err != nil {
		return empty, err
	}
	prepared, err := auditdomain.BuildPreparedOpenAPIWork(ctx, input)
	if err != nil {
		return empty, err
	}
	list := auditcontract.PreparedWork{Schema: auditcontract.PreparedWorkSchema, Work: make([]auditcontract.PreparedSelection, 0, len(prepared.Items)), Gaps: prepared.Gaps}
	for _, item := range prepared.Items {
		if err := ctx.Err(); err != nil {
			return empty, err
		}
		name := "prepared-openapi-" + strings.TrimPrefix(auditdomain.DigestBytes(item.Data), "sha256:")
		if name == output.Name {
			return empty, invalid()
		}
		material, err := m.publish(ctx, call, contracts.ArtifactRef{Namespace: output.Namespace, Name: name}, item.Data)
		if err != nil {
			return empty, err
		}
		selected := map[string]auditcontract.Material{"operation": material, "openapi": input.Source}
		if input.Settings != nil {
			selected["settings"] = *input.Settings
		}
		list.Work = append(list.Work, auditcontract.PreparedSelection{Key: item.Key, Workflow: options.Workflow, Objective: item.Objective, Inputs: selected})
	}
	data, err := strictjson.Canonical(list)
	if err != nil {
		return empty, err
	}
	data, err = auditcontract.Decode(data, nil)
	if err != nil {
		return empty, invalid()
	}
	root, err := m.publish(ctx, call, output, data)
	if err != nil {
		return empty, err
	}
	return stageresult.StageContentResult{APIVersion: contracts.APIVersion, Outcome: stageresult.StageSucceeded, Summary: fmt.Sprintf("Prepared %d independently selectable OpenAPI operations with %d retained preparation gaps.", len(list.Work), len(list.Gaps)), Artifacts: map[string]contracts.ArtifactRef{"work": root.Ref}}, nil
}

func descriptor(runID string, metadata artifacts.Metadata) auditcontract.Material {
	return auditcontract.Material{ScopeKind: "run", ScopeID: runID, Ref: metadata.Ref.Clone(), Digest: metadata.Digest, MediaType: metadata.MediaType, SizeBytes: metadata.Size}
}
