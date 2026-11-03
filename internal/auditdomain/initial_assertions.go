package auditdomain

import (
	"encoding/json"
	"fmt"
	"github.com/grauwolf32/contractor/internal/auditstandards"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	"maps"
	"slices"
	"strings"
)

// ChecklistAssertions interprets a finite authored source through the owning
// domain's strict JSON/YAML parser. It creates content, never identities or Runs.
func ChecklistAssertions(data []byte, mediaType, workflow string) (auditcontract.InitialAssertions, error) {
	empty := auditcontract.InitialAssertions{}
	if len(data) > auditcontract.MaxMutationBytes {
		return empty, invalid(CodeLimitExceeded, "checklist")
	}
	value, err := parseJSONOrYAML(data, mediaType)
	if err != nil {
		return empty, err
	}
	canonical, err := strictjson.Canonical(value)
	if err != nil {
		return empty, err
	}
	canonical, err = auditcontract.Decode(canonical, nil)
	if err != nil {
		return empty, err
	}
	var source struct {
		Schema string `json:"schema"`
		Items  []struct {
			Version          string            `json:"version"`
			Applicability    string            `json:"applicability"`
			AllowedMethods   []string          `json:"allowed_methods"`
			RequiredEvidence []string          `json:"required_evidence"`
			Key              string            `json:"key"`
			Statement        string            `json:"statement"`
			Scope            map[string]string `json:"scope"`
			Grounds          string            `json:"grounds"`
		} `json:"items"`
	}
	if strictjson.Decode(canonical, &source) != nil || source.Schema != "contractor.audit.assertions.v1" || len(source.Items) == 0 || len(source.Items) > 128 {
		return empty, invalid(CodeInvalid, "checklist")
	}
	plan := auditcontract.InitialAssertions{Hypotheses: map[string]auditcontract.HypothesisSeed{}, Work: []auditcontract.InitialWork{}}
	for _, entry := range source.Items {
		if _, exists := plan.Hypotheses[entry.Key]; exists {
			return empty, invalid(CodeInvalid, "checklist.key")
		}
		if !contracts.ValidConfigVersion(entry.Version) || strings.TrimSpace(entry.Applicability) == "" || entry.Scope == nil || len(entry.AllowedMethods) == 0 || entry.RequiredEvidence == nil {
			return empty, invalid(CodeInvalid, "checklist.source")
		}
		if validateSortedStrings(entry.AllowedMethods, MaximumCoverageValues, "allowed_methods", true) != nil || validateSortedStrings(entry.RequiredEvidence, MaximumEvidencePerItem, "required_evidence", false) != nil {
			return empty, invalid(CodeInvalid, "checklist.evidence")
		}
		scope := maps.Clone(entry.Scope)
		scope["source_version"] = entry.Version
		scope["applicability"] = entry.Applicability
		methods, _ := strictjson.Canonical(entry.AllowedMethods)
		evidence, _ := strictjson.Canonical(entry.RequiredEvidence)
		scope["allowed_methods"] = string(methods)
		scope["required_evidence"] = string(evidence)
		plan.Hypotheses[entry.Key] = auditcontract.HypothesisSeed{Statement: entry.Statement, Scope: scope, Grounds: entry.Grounds}
		data, err := strictjson.Canonical(plan.Hypotheses[entry.Key])
		if err != nil {
			return empty, err
		}
		if _, err := auditcontract.DecodeHypothesisSeed(data); err != nil {
			return empty, err
		}
		plan.Work = append(plan.Work, auditcontract.InitialWork{Key: entry.Key, Workflow: workflow, Hypothesis: entry.Key, Objective: entry.Statement})
	}
	return plan, nil
}

// StandardAssertions preserves each selected entry, its applicability and exact
// evidence sufficiency contract. Mapping availability never changes coverage.
func StandardAssertions(pkg auditstandards.Package, selection auditcontract.SeedSource) (auditcontract.InitialAssertions, error) {
	plan := auditcontract.InitialAssertions{Hypotheses: map[string]auditcontract.HypothesisSeed{}, Work: []auditcontract.InitialWork{}}
	entries, err := auditstandards.SelectEntries(pkg, selection.EntryIDs, auditcontract.MaxMutationRecords)
	if err != nil {
		return auditcontract.InitialAssertions{}, err
	}
	evidenceContracts := map[string]auditstandards.EvidenceContract{}
	for _, c := range pkg.Document.EvidenceContracts {
		evidenceContracts[c.ID+"@"+c.Version] = c
	}
	mappings := map[string][]auditstandards.Mapping{}
	for _, mapping := range pkg.Document.Mappings {
		for _, id := range mapping.EntryIDs {
			mappings[id] = append(mappings[id], mapping)
		}
	}
	for _, entry := range entries {
		key := fmt.Sprintf("standard-%03d", len(plan.Hypotheses)+1)
		evidence, err := strictjson.Canonical(evidenceContracts[entry.EvidenceContract.ID+"@"+entry.EvidenceContract.Version])
		if err != nil {
			return plan, err
		}
		mappingJSON, err := strictjson.Canonical(mappings[entry.ID])
		if err != nil {
			return plan, err
		}
		methods, _ := strictjson.Canonical(entry.AllowedMethods)
		plan.Hypotheses[key] = auditcontract.HypothesisSeed{Statement: entry.Statement, Scope: map[string]string{"standard_mapping": string(mappingJSON), "standard_scheme": pkg.Reference().Scheme, "standard_version": pkg.Reference().Version, "entry_id": entry.ID, "applicability": entry.Applicability.Mode, "applicability_rule": entry.Applicability.Rule, "allowed_methods": string(methods), "evidence_contract": string(evidence)}, Grounds: pkg.Document.Standard.Title + ": " + entry.Title}
		// Empty optional scope values are omitted: common text is always nonblank.
		for k, v := range plan.Hypotheses[key].Scope {
			if v == "" {
				delete(plan.Hypotheses[key].Scope, k)
			}
		}
		mapped := false
		for _, mapping := range mappings[entry.ID] {
			mapped = mapped || mapping.WorkflowRole == selection.Workflow
		}
		if mapped {
			plan.Work = append(plan.Work, auditcontract.InitialWork{Key: key, Workflow: selection.Workflow, Hypothesis: key, Objective: entry.Statement})
		} else {
			plan.Hypotheses[key].Scope["coverage_gap"] = "No selected checking mapping applies; requirement remains untested"
		}
	}
	if len(plan.Hypotheses) == 0 || len(plan.Hypotheses) > auditcontract.MaxMutationRecords {
		return auditcontract.InitialAssertions{}, invalid(CodeInvalid, "standard.selection")
	}
	return plan, nil
}

// ExecutionProgram overlays only a bounded Start-frozen assertion selection.
// The authored catalog identity and Workflow closures retain their original pin.
func ValidateInitialAssertions(plan auditcontract.InitialAssertions) error {
	data, err := strictjson.Canonical(plan)
	source := plan.Source
	if err != nil || len(data) > auditcontract.MaxMutationBytes || len(plan.Hypotheses) == 0 || len(plan.Hypotheses) > auditcontract.MaxMutationRecords || plan.Work == nil || len(plan.Work) > len(plan.Hypotheses) || source.ScopeKind != "audit" || !contracts.ValidResourceID(source.ScopeID) || source.Ref.ValidateExact() != nil || source.Ref.Namespace != "inputs" || source.Ref.Name != "assertion-source" || !contentdigest.Valid(source.Digest) || source.SizeBytes < 1 || source.SizeBytes > auditcontract.MaximumDocumentBytes || !contracts.ValidMediaType(source.MediaType) {
		return invalid(CodeInvalid, "initial_assertions")
	}
	for key, seed := range plan.Hypotheses {
		data, err := strictjson.Canonical(seed)
		if !contracts.ValidName(key) || err != nil {
			return invalid(CodeInvalid, "initial_assertions")
		}
		if _, err := auditcontract.DecodeHypothesisSeed(data); err != nil {
			return err
		}
	}
	return nil
}

func ExecutionProgram(program auditcontract.Program, baseline []byte) (auditcontract.Program, error) {
	var fields map[string]json.RawMessage
	if strictjson.Decode(baseline, &fields) != nil {
		return auditcontract.Program{}, invalid(CodeInvalid, "baseline")
	}
	raw := fields["initial_assertions"]
	if raw == nil {
		return program, nil
	}
	var plan auditcontract.InitialAssertions
	if len(raw) > auditcontract.MaxMutationBytes || strictjson.Decode(raw, &plan) != nil || program.SeedSource == nil || ValidateInitialAssertions(plan) != nil {
		return auditcontract.Program{}, invalid(CodeInvalid, "initial_assertions")
	}
	program.Hypotheses = maps.Clone(program.Hypotheses)
	program.InitialWork = slices.Clone(program.InitialWork)
	for key, seed := range plan.Hypotheses {
		if _, exists := program.Hypotheses[key]; exists {
			return auditcontract.Program{}, invalid(CodeInvalid, "seed.collision")
		}
		program.Hypotheses[key] = seed
	}
	for _, work := range plan.Work {
		if work.Workflow != program.SeedSource.Workflow {
			return auditcontract.Program{}, invalid(CodeInvalid, "seed.workflow")
		}
		program.InitialWork = append(program.InitialWork, work)
	}
	data, err := strictjson.Canonical(program)
	if err != nil {
		return auditcontract.Program{}, err
	}
	return auditcontract.DecodeProgram(data)
}
