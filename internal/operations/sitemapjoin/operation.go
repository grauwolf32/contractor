// Package sitemapjoin adapts declared Stage inputs to sitemap domain services.
// Parsing, joining, publication and receipts belong to projectsitemap.
package sitemapjoin

import (
	"context"
	"errors"
	"fmt"

	"github.com/grauwolf32/contractor/internal/apierror"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/operations"
	"github.com/grauwolf32/contractor/internal/projectsitemap"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/stageresult"
)

type Module struct {
	service *projectsitemap.Service
	apply   bool
}

func NewJoin(service *projectsitemap.Service) *Module  { return &Module{service: service} }
func NewApply(service *projectsitemap.Service) *Module { return &Module{service: service, apply: true} }

func (m *Module) Ref() string {
	if m.apply {
		return contracts.SitemapApplyOpenAPIOperation
	}
	return contracts.SitemapOpenAPIJoinOperation
}

func (m *Module) Run(ctx context.Context, call operations.Call) (stageresult.StageContentResult, error) {
	invalid := func() (stageresult.StageContentResult, error) {
		return stageresult.StageContentResult{}, &operations.Failure{Code: "invalid_request", Message: "Sitemap operation inputs, arguments or outputs are invalid"}
	}
	if len(call.Inputs) != 2 || call.Inputs["sitemap"].ValidateExact() != nil || call.Inputs["openapi"].ValidateExact() != nil || len(call.Outputs) > 2 {
		return invalid()
	}
	for name := range call.Outputs {
		if name != "sitemap" && name != "report" {
			return invalid()
		}
	}
	for name := range call.Arguments {
		if name != "origin" && name != "contract" && name != "deployment" {
			return invalid()
		}
	}
	origin, ok := call.Arguments["origin"].(string)
	if !ok {
		return invalid()
	}
	contract, ok := call.Arguments["contract"].(string)
	if !ok {
		return invalid()
	}
	deployment := ""
	if value, exists := call.Arguments["deployment"]; exists {
		var ok bool
		deployment, ok = value.(string)
		if !ok {
			return invalid()
		}
	}
	request := projectsitemap.StageJoin{RunID: call.RunID, StageID: call.StageExecutionID, ClaimID: call.ClaimID, Sitemap: call.Inputs["sitemap"], OpenAPI: call.Inputs["openapi"], Options: projectsitemap.JoinOptions{Origin: origin, Contract: contract, Deployment: deployment}}
	if output, exists := call.Outputs["sitemap"]; exists {
		request.Output = &output
	}
	if report, exists := call.Outputs["report"]; exists {
		request.Report = &report
	}
	var result projectsitemap.StageJoinResult
	var err error
	if m.apply {
		result, err = m.service.ApplyContractStage(ctx, request)
	} else {
		result, err = m.service.JoinStage(ctx, request)
	}
	if err != nil {
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) || errors.Is(err, runstore.ErrConflict) {
			return stageresult.StageContentResult{}, err
		}
		if classified, ok := apierror.Of(err); ok {
			kind := classified.ErrorKind()
			return stageresult.StageContentResult{}, &operations.Failure{Code: kind.Code(), Message: kind.Message(), Retryable: kind.Retryable()}
		}
		return stageresult.StageContentResult{}, err
	}
	outputs := map[string]contracts.ArtifactRef{}
	if result.Output != nil {
		outputs["sitemap"] = *result.Output
	}
	if result.Report != nil {
		outputs["report"] = *result.Report
	}
	summary := fmt.Sprintf("Compared %d declared operations and %d retained observed members; %d matched, %d declared-only, %d observed-only, %d ambiguous members, %d unresolved declarations.", result.Summary.Declared, result.Summary.Observed, result.Summary.Matched, result.Summary.DeclaredOnly, result.Summary.ObservedOnly, result.Summary.Ambiguous, result.Summary.Unresolved)
	if result.Publication != nil {
		summary += fmt.Sprintf(" Published Project %s/%s@%s.", result.Publication.Namespace, result.Publication.Name, *result.Publication.Revision)
	}
	return stageresult.StageContentResult{APIVersion: contracts.APIVersion, Outcome: stageresult.StageSucceeded, Summary: summary, Artifacts: outputs}, nil
}
