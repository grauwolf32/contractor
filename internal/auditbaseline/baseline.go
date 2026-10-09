// Package auditbaseline owns the immutable Audit baseline wire identity and
// data projections shared by the start service and downstream readers.
package auditbaseline

import (
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstandards"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
)

const Schema = "contractor.audit.baseline.v1"
const maximumScopeFields = 3

type Scope struct {
	Objective          string `json:"objective,omitempty"`
	Target             string `json:"target,omitempty"`
	AuthorizationScope string `json:"authorizationScope,omitempty"`
}

func (s Scope) Values() map[string]string {
	result := make(map[string]string, maximumScopeFields)
	if s.Objective != "" {
		result["objective"] = s.Objective
	}
	if s.Target != "" {
		result["target"] = s.Target
	}
	if s.AuthorizationScope != "" {
		result["authorizationScope"] = s.AuthorizationScope
	}
	return result
}

type BaselineInventory struct {
	SourceContentDigest      string                         `json:"sourceContentDigest"`
	CanonicalInventoryDigest string                         `json:"canonicalInventoryDigest"`
	StandardSelection        *config.AuditStandardSelection `json:"standardSelection,omitempty"`
	Gaps                     []string                       `json:"gaps"`
	Worklist                 auditstore.ExactArtifact       `json:"worklist"`
	ExecutionManifest        auditdomain.ExecutionManifest  `json:"executionManifest"`
}

const DerivedInventorySchema = "contractor.audit.derived-inventory.v1"

// DerivedInventory is retained separately from the original immutable baseline.
// Preparation outputs retain both their accepted receipt and exact source Run.
type DerivedInventory struct {
	Schema    string                              `json:"schema"`
	Inventory BaselineInventory                   `json:"inventory"`
	Sources   map[string]auditstore.ExactArtifact `json:"sources"`
}

type BaselineSnapshot struct {
	Schema               string                              `json:"schema"`
	Inputs               map[string]auditstore.ExactArtifact `json:"inputs"`
	Scope                Scope                               `json:"scope"`
	RuntimeLabels        []string                            `json:"runtimeLabels"`
	RuntimeConfig        runtimeconfig.RunSnapshot           `json:"runtimeConfig"`
	Skills               []contracts.RunSkillSnapshot        `json:"skills"`
	LLMCredentialIDs     []string                            `json:"llmCredentialIds"`
	RuntimeCredentialIDs []string                            `json:"runtimeCredentialIds"`
	ProjectHTTPTarget    *contracts.HTTPOriginTargetRef      `json:"projectHttpTarget,omitempty"`
	Standards            []auditstandards.PinnedPackage      `json:"standards"`
	Inventory            *BaselineInventory                  `json:"inventory,omitempty"`
}
