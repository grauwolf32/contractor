package auditbaseline

import (
	"bytes"
	"encoding/json"
	"fmt"
	"maps"
	"slices"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/ucarion/jcs"
)

type CheckRoutePins struct {
	Route         auditdomain.AcceptedPlanRoute
	Authorization auditdomain.AcceptedPlanAuthorization
}

// CheckRoutes compiles one exact profile/baseline closure for admission
// and dispatch. Names select bindings only through their registered semantic
// method key; role spelling cannot classify a check or broaden its authority.
type CheckRoutes struct {
	audit    auditstore.Audit
	bindings map[contracts.CheckPlanKey]CheckRoutePins
}

func NewCheckRoutes(audit auditstore.Audit, profile auditprofile.ResolvedAuditProfile, baseline BaselineSnapshot) (*CheckRoutes, error) {
	if baseline.Schema != Schema || profile.Ref.Digest != audit.Profile.Digest || profile.Ref.Name != audit.Profile.Name || profile.Ref.Version != audit.Profile.Version || !baseline.ActiveChecks.Valid() {
		return nil, fmt.Errorf("%w: typed check profile or baseline pins are invalid", auditstore.ErrInvalid)
	}
	methods, err := auditprofile.CompileCheckMethods(profile)
	if err != nil {
		return nil, fmt.Errorf("%w: typed check method mapping is invalid: %v", auditstore.ErrInvalid, err)
	}
	scopeDigest, err := CheckAdmissionDigest(baseline.Scope)
	if err != nil {
		return nil, err
	}
	limitsDigest, err := CheckAdmissionDigest(audit.Limits)
	if err != nil {
		return nil, err
	}
	result := &CheckRoutes{audit: audit, bindings: map[contracts.CheckPlanKey]CheckRoutePins{}}
	consumers := map[string]struct{ binding, workflow, skills string }{}
	for _, name := range slices.Sorted(maps.Keys(profile.Methods)) {
		mapping := profile.Methods[name]
		key, known := contracts.LookupCheckPlanMethod(mapping.PlanSchema, mapping.Method)
		if !known {
			return nil, fmt.Errorf("%w: typed check method is unregistered", auditstore.ErrInvalid)
		}
		binding := profile.Workflows[mapping.WorkflowRole]
		pins, found := consumers[mapping.WorkflowRole]
		if !found {
			pins.binding, err = CheckAdmissionDigest(binding)
			if err != nil {
				return nil, err
			}
			// These two opaque digests intentionally preserve the V62-006 wire
			// bytes rather than reinterpreting config snapshots as plan JSON.
			pins.workflow, pins.skills, err = ConsumerContentPins(binding, baseline)
			if err != nil {
				return nil, err
			}
			consumers[mapping.WorkflowRole] = pins
		}
		mappingDigest, err := CheckAdmissionDigest(mapping)
		if err != nil {
			return nil, err
		}
		approval := auditdomain.ApprovalNone
		active := auditprofile.WorkflowRoleSelectsClassifiedTool(profile, mapping.WorkflowRole, true)
		if active && baseline.ActiveChecks == contracts.ActiveChecksProhibited {
			return nil, fmt.Errorf("%w: typed check requires prohibited active tools", auditstore.ErrInvalid)
		}
		if active && baseline.ActiveChecks == contracts.ActiveChecksApprovalRequired {
			approval = auditdomain.ApprovalActiveCheck
		}
		selected, _ := methods.Select(key)
		result.bindings[key] = CheckRoutePins{Route: auditdomain.AcceptedPlanRoute{MappingID: name, MappingDigest: mappingDigest, WorkflowRole: mapping.WorkflowRole, BindingDigest: pins.binding, WorkflowDigest: pins.workflow, SkillsDigest: pins.skills, RequiredCapabilities: selected.Mapping.RequiredCapabilities}, Authorization: auditdomain.AcceptedPlanAuthorization{ScopeDigest: scopeDigest, ActiveChecks: baseline.ActiveChecks, ApprovalRequirement: approval, LimitsDigest: limitsDigest}}
	}
	return result, nil
}

func (c *CheckRoutes) Select(plan auditdomain.CheckPlan) (CheckRoutePins, bool) {
	key := contracts.CheckPlanKey{Schema: plan.Schema, Method: plan.Method, Phase: plan.Phase}
	value, ok := c.bindings[key]
	value.Route.RequiredCapabilities = slices.Clone(value.Route.RequiredCapabilities)
	for index := range value.Route.RequiredCapabilities {
		value.Route.RequiredCapabilities[index].Tools = slices.Clone(value.Route.RequiredCapabilities[index].Tools)
	}
	return value, ok
}

func (c *CheckRoutes) Build(source auditdomain.AcceptedPlanSource, plan auditdomain.CheckPlan, deps *auditdomain.RetainedDependencyManifest) (auditdomain.AcceptedPlan, error) {
	binding, ok := c.Select(plan)
	if !ok {
		return auditdomain.AcceptedPlan{}, fmt.Errorf("%w: typed check method has no configured route", auditstore.ErrInvalid)
	}
	if source.ProposedCheckOrdinal == nil {
		return auditdomain.AcceptedPlan{}, fmt.Errorf("%w: typed check source ordinal is absent", auditstore.ErrInvalid)
	}
	ordinal := *source.ProposedCheckOrdinal
	source.ProposedCheckOrdinal = &ordinal
	source.Proposal = source.Proposal.Clone()
	source.RetainedProposal = source.RetainedProposal.Clone()
	encoded, err := auditdomain.EncodeCheckPlan(plan)
	if err != nil {
		return auditdomain.AcceptedPlan{}, err
	}
	accepted := auditdomain.AcceptedPlan{Schema: auditdomain.AcceptedPlanSchema, OwnerID: c.audit.OwnerID, ProjectID: c.audit.ProjectID, AuditID: c.audit.AuditID, ItemKey: auditdomain.FindingCheckItemKey(source.ReceiptID, ordinal), BaselineDigest: auditdomain.DigestBytes(c.audit.BaselineSnapshot), ProfileDigest: c.audit.Profile.Digest, Source: source, Plan: plan, PlanDigest: auditdomain.DigestBytes(encoded), Route: binding.Route, Authorization: binding.Authorization}
	if deps != nil {
		encoded, err := auditdomain.EncodeRetainedDependencies(*deps)
		if err != nil {
			return auditdomain.AcceptedPlan{}, err
		}
		digest := auditdomain.DigestBytes(encoded)
		accepted.DependenciesDigest = &digest
	}
	return accepted, auditdomain.ValidateAcceptedPlan(accepted, deps)
}

// Validate checks immutable source, route and authorization pins against the
// actual consumer closure. PRJ-R104 deliberately pins the effective Project
// policy at Audit start; a later Project edit does not rewrite this baseline.
// Current owner/Project lifecycle and exact item approval remain store fences.
func (c *CheckRoutes) Validate(plan auditdomain.AcceptedPlan, deps *auditdomain.RetainedDependencyManifest, workflowRole string) error {
	expected, err := c.Build(plan.Source, plan.Plan, deps)
	if err != nil {
		return err
	}
	if workflowRole != expected.Route.WorkflowRole {
		return fmt.Errorf("%w: typed check Workflow role differs from its selected route", auditstore.ErrInvalid)
	}
	actualBytes, err := auditdomain.EncodeAcceptedPlan(plan, deps)
	if err != nil {
		return err
	}
	expectedBytes, err := auditdomain.EncodeAcceptedPlan(expected, deps)
	if err != nil {
		return err
	}
	if !bytes.Equal(actualBytes, expectedBytes) {
		return fmt.Errorf("%w: typed check source, route or authorization pins drifted", auditstore.ErrInvalid)
	}
	return nil
}

func CheckAdmissionDigest(value any) (string, error) {
	encoded, err := json.Marshal(value)
	if err != nil {
		return "", err
	}
	var document any
	if err := json.Unmarshal(encoded, &document); err != nil {
		return "", err
	}
	canonical, err := jcs.Format(document)
	if err != nil {
		return "", err
	}
	return auditdomain.DigestBytes([]byte(canonical)), nil
}
