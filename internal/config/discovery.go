package config

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"maps"
	"slices"
	"sort"

	"github.com/grauwolf32/contractor/internal/config/resolved"
)

// WorkflowDiscovery is the Workflow discovery view of one immutable Snapshot:
// its Workflows in exact selector order and the SHA-256 fingerprint of their
// encoding that binds a discovery cursor (CAT-R67). The Snapshot builds both
// once, so a page read neither copies nor encodes the catalog.
type WorkflowDiscovery struct {
	SourceFingerprint string
	selectors         []string
	workflows         []resolved.ResolvedWorkflow
}

// Page returns caller-owned copies of at most limit Workflows whose exact
// selector sorts after after and that matches accepts. matches sees only the
// identity and the authored presentation of a candidate; an absent
// presentation is the zero value.
func (d WorkflowDiscovery) Page(
	after string, limit int, matches func(resolved.WorkflowRef, resolved.WorkflowPresentation) bool,
) []resolved.ResolvedWorkflow {
	start := sort.Search(len(d.selectors), func(index int) bool { return d.selectors[index] > after })
	result := make([]resolved.ResolvedWorkflow, 0, min(max(limit, 0), len(d.workflows)-start))
	for _, workflow := range d.workflows[start:] {
		if len(result) >= limit {
			break
		}
		var presentation resolved.WorkflowPresentation
		if workflow.Presentation != nil {
			presentation = *workflow.Presentation
		}
		if matches(workflow.Ref, presentation) {
			result = append(result, workflow.Clone())
		}
	}
	return result
}

// ConfigurationDiscovery is one configuration kind's discovery view of an
// immutable Snapshot: caller-owned resources in exact selector order and the
// SHA-256 fingerprint of their encoding that binds a discovery cursor
// (CAT-R67). The fingerprint is computed once, when the Snapshot is built.
type ConfigurationDiscovery struct {
	SourceFingerprint string
	Resources         []ConfigurationResource
}

// configurationKinds lists every kind ParseConfigurationKind accepts.
var configurationKinds = []ConfigurationKind{
	ConfigurationAgentTemplates, ConfigurationExecutionConfigs,
	ConfigurationModelPolicies, ConfigurationLLMGateways,
}

// catalogFingerprint is the result of encoding one catalog source once.
type catalogFingerprint struct {
	value string
	err   error
}

func fingerprintCatalogSource(encoded []byte, err error) catalogFingerprint {
	if err != nil {
		return catalogFingerprint{err: err}
	}
	digest := sha256.Sum256(encoded)
	return catalogFingerprint{value: hex.EncodeToString(digest[:])}
}

func (f catalogFingerprint) get() (string, error) {
	if f.err != nil {
		return "", fmt.Errorf("fingerprint catalog source: %w", f.err)
	}
	return f.value, nil
}

// buildDiscovery encodes every discovery source once while the Snapshot is
// built, never on a discovery request, and returns the encoded Workflow list
// that the AgentTemplate usage index fingerprints too.
func (s *Snapshot) buildDiscovery() ([]byte, error) {
	selectors := slices.Sorted(maps.Keys(s.workflows))
	workflows := make([]resolved.ResolvedWorkflow, 0, len(selectors))
	for _, selector := range selectors {
		workflows = append(workflows, s.workflows[selector])
	}
	workflowJSON, err := json.Marshal(workflows)
	s.workflowDiscovery = WorkflowDiscovery{selectors: selectors, workflows: workflows}
	s.workflowFingerprint = fingerprintCatalogSource(workflowJSON, err)
	s.configurationFingerprints = make(map[ConfigurationKind]catalogFingerprint, len(configurationKinds))
	for _, kind := range configurationKinds {
		resources, resourceErr := s.configurations(kind)
		if resourceErr != nil {
			s.configurationFingerprints[kind] = catalogFingerprint{err: resourceErr}
			continue
		}
		s.configurationFingerprints[kind] = fingerprintCatalogSource(json.Marshal(resources))
	}
	return workflowJSON, err
}

// WorkflowDiscovery returns the snapshot's Workflow discovery view.
func (s *Snapshot) WorkflowDiscovery() (WorkflowDiscovery, error) {
	fingerprint, err := s.workflowFingerprint.get()
	if err != nil {
		return WorkflowDiscovery{}, err
	}
	discovery := s.workflowDiscovery
	discovery.SourceFingerprint = fingerprint
	return discovery, nil
}

// ConfigurationDiscovery returns kind's resources and their fingerprint, both
// from this one snapshot.
func (s *Snapshot) ConfigurationDiscovery(kind ConfigurationKind) (ConfigurationDiscovery, error) {
	if _, err := ParseConfigurationKind(string(kind)); err != nil {
		return ConfigurationDiscovery{}, err
	}
	fingerprint, err := s.configurationFingerprints[kind].get()
	if err != nil {
		return ConfigurationDiscovery{}, err
	}
	resources, err := s.configurations(kind)
	if err != nil {
		return ConfigurationDiscovery{}, err
	}
	return ConfigurationDiscovery{SourceFingerprint: fingerprint, Resources: resources}, nil
}
