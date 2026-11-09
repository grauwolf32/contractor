// Package auditbaseline owns the immutable Audit baseline wire identity and
// data projections shared by the start service and downstream readers.
package auditbaseline

import (
	"github.com/grauwolf32/contractor/internal/agentskills"
	"github.com/grauwolf32/contractor/internal/auditstandards"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	"github.com/grauwolf32/contractor/internal/contracts/runlabels"
	"github.com/grauwolf32/contractor/internal/contracts/runtimesettings"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
)

const Schema = "contractor.audit.baseline.v1"

type Scope struct {
	Objective          string `json:"objective,omitempty"`
	Target             string `json:"target,omitempty"`
	AuthorizationScope string `json:"authorizationScope,omitempty"`
}

type BaselineSnapshot struct {
	Parameters              map[string]string                    `json:"parameters,omitempty"`
	InitialAssertions       *auditcontract.InitialAssertions     `json:"initial_assertions,omitempty"`
	Engagement              *EngagementSnapshot                  `json:"engagement,omitempty"`
	Labels                  runlabels.RunMetadataLabels          `json:"labels"`
	RequiredPlacementLabels []string                             `json:"requiredPlacementLabels"`
	Schema                  string                               `json:"schema"`
	ActiveChecks            contracts.ActiveChecksPolicy         `json:"activeChecks"`
	Inputs                  map[string]auditstore.ExactArtifact  `json:"inputs"`
	Scope                   Scope                                `json:"scope"`
	RuntimeLabels           []string                             `json:"runtimeLabels"`
	RuntimeConfig           runtimeconfig.RunSnapshot            `json:"runtimeConfig"`
	Skills                  []agentskills.RunSkillSnapshot       `json:"skills"`
	LLMCredentialIDs        []string                             `json:"llmCredentialIds"`
	RuntimeCredentialIDs    []string                             `json:"runtimeCredentialIds"`
	ProjectHTTPTarget       *runtimesettings.HTTPOriginTargetRef `json:"projectHttpTarget,omitempty"`
	Standards               []auditstandards.PinnedPackage       `json:"standards"`
}
