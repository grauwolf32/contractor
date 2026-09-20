package evalservice

import (
	"context"
	"errors"
	"sort"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/grauwolf32/contractor/internal/evalstore"
	pg "github.com/grauwolf32/contractor/internal/persistence/postgres"
)

type nativeChecker struct {
	db             pg.DBTX
	result         evaldomain.ResultInput
	outputs        map[string]evaldomain.Output
	evidence       map[evaldomain.Artifact][]string
	remainingBytes int64
}

func runNativeChecks(ctx context.Context, db pg.DBTX, checks []evaldomain.Check, outputs map[string]evaldomain.Output, result evaldomain.ResultInput) ([]evaldomain.CheckResult, error) {
	checker := nativeChecker{
		db:             db,
		result:         result,
		outputs:        outputs,
		evidence:       map[evaldomain.Artifact][]string{},
		remainingBytes: evaldomain.MaxCollectionBytes,
	}
	for _, ref := range result.Evidence {
		checker.evidence[ref.Artifact] = append(checker.evidence[ref.Artifact], ref.ID)
	}
	results := make([]evaldomain.CheckResult, 0, len(checks))
	for _, check := range checks {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		result, err := checker.check(ctx, check)
		if err != nil {
			return nil, err
		}
		results = append(results, result)
	}
	return results, nil
}

func (c *nativeChecker) roles(check evaldomain.Check) []string {
	if role := check.Parameters["output"]; role != "" {
		return []string{role}
	}
	roles := []string{}
	for role, contract := range c.outputs {
		if contract.Required {
			roles = append(roles, role)
		}
	}
	sort.Strings(roles)
	return roles
}

func (c *nativeChecker) check(ctx context.Context, check evaldomain.Check) (evaldomain.CheckResult, error) {
	result := evaldomain.CheckResult{
		ID:                   check.ID,
		Evaluator:            check.Evaluator,
		ImplementationSHA256: check.ImplementationSHA256,
		Status:               "pass",
		EvidenceRefs:         []string{},
	}
	if check.Evaluator == "human-review@1" {
		result.Status, result.Reason = "incomplete", "An authenticated owner review over this exact result is required."
		return result, nil
	}
	if err := validateNativeCheck(check); err != nil {
		result.Status, result.Reason = "error", "Pinned native evaluator is unavailable."
		return result, nil
	}
	roles := c.roles(check)
	if len(roles) == 0 {
		result.Status, result.Reason = "incomplete", "No required output contract was selected."
		return result, nil
	}
	for _, role := range roles {
		status, reason, refs, err := c.checkOutput(ctx, check, role)
		if err != nil {
			return result, err
		}
		if nativeCheckSeverity(status) > nativeCheckSeverity(result.Status) {
			result.Status, result.Reason = status, reason
		}
		result.EvidenceRefs = append(result.EvidenceRefs, refs...)
	}
	result.EvidenceRefs = sortedUnique(result.EvidenceRefs)
	return result, nil
}

func nativeCheckSeverity(status string) int {
	switch status {
	case "error":
		return 3
	case "fail":
		return 2
	case "incomplete":
		return 1
	default:
		return 0
	}
}

func (c *nativeChecker) checkOutput(ctx context.Context, check evaldomain.Check, role string) (string, string, []string, error) {
	ref, exists := c.result.Outputs[role]
	if !exists {
		return "incomplete", "Required output evidence is unavailable.", nil, nil
	}
	if err := evalstore.NewPostgresStore(c.db).VerifyEvidence(ctx, ref); err != nil {
		if evaldomain.IsCode(err, "eval_evidence_unavailable") {
			return "incomplete", "Exact output evidence is unavailable.", nil, nil
		}
		if evaldomain.IsCode(err, "eval_member_conflict") {
			return "error", "Exact output integrity verification failed.", nil, nil
		}
		return "", "", nil, err
	}
	refs := c.evidence[ref]
	if check.Evaluator == "media-type@1" && ref.MediaType != check.Parameters["mediaType"] {
		return "fail", "Output media type does not match the declared media type.", refs, nil
	}
	if check.Evaluator != "json-schema@1" {
		return "pass", "", refs, nil
	}
	status, reason, err := c.validateJSON(ctx, check, ref)
	return status, reason, refs, err
}

func (c *nativeChecker) validateJSON(ctx context.Context, check evaldomain.Check, ref evaldomain.Artifact) (string, string, error) {
	if ref.SizeBytes > evaldomain.MaxDocumentBytes || ref.SizeBytes > c.remainingBytes {
		return "incomplete", "Output exceeds the registered validator's byte bound.", nil
	}
	c.remainingBytes -= ref.SizeBytes
	scope, err := nativeEvidenceScope(ref)
	if err != nil {
		return "", "", err
	}
	read, err := artifacts.NewPostgresRepository(c.db).Read(ctx, scope, contracts.ArtifactRef{Namespace: ref.Namespace, Name: ref.Name, Revision: &ref.Revision})
	if errors.Is(err, artifacts.ErrArtifactNotFound) || errors.Is(err, artifacts.ErrBlobMissing) {
		return "incomplete", "Exact output bytes are unavailable.", nil
	}
	if errors.Is(err, artifacts.ErrArtifactIntegrity) {
		return "error", "Exact output integrity verification failed.", nil
	}
	if err != nil {
		return "", "", err
	}
	if evaldomain.Digest(read.Payload.Data) != ref.SHA256 {
		return "error", "Exact output integrity verification failed.", nil
	}
	if err = validateRegisteredSchema(check.Parameters["schema"], read.Payload.Data); err != nil {
		return "fail", "Output does not satisfy the selected structural schema.", nil
	}
	return "pass", "", nil
}

func nativeEvidenceScope(ref evaldomain.Artifact) (artifacts.Scope, error) {
	switch ref.Scope {
	case "run":
		return artifacts.RunScope(ref.ScopeID)
	case "project":
		return artifacts.ProjectScope(ref.ScopeID)
	case "user":
		return artifacts.UserScope(ref.ScopeID)
	default:
		return artifacts.Scope{}, evaldomain.Failure("eval_invalid")
	}
}
