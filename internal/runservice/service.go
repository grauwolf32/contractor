package runservice

import (
	"fmt"

	"github.com/grauwolf32/contractor/internal/contracts"
)

func New(options Options) (*Service, error) {
	if options.Runs == nil || options.Workflows == nil || options.LLMCredentials == nil ||
		options.CredentialGuard == nil || options.RuntimeCredentials == nil ||
		options.Projects == nil || options.PublicTransaction == nil {
		return nil, fmt.Errorf("%w: public Run dependencies are incomplete", ErrNotConfigured)
	}
	return &Service{
		runs: options.Runs, workflows: options.Workflows,
		llmCredentials: options.LLMCredentials, credentialGuard: options.CredentialGuard,
		runtimeCredentials: options.RuntimeCredentials, projects: options.Projects,
		publicTransaction: options.PublicTransaction, auditTransaction: options.AuditTransaction,
		skillInitializationAvailable: options.SkillInitializationAvailable,
	}, nil
}

func cloneParameters(source map[string]string) map[string]string {
	result := make(map[string]string, len(source))
	for key, value := range source {
		result[key] = value
	}
	return result
}

func cloneRefs(source map[string]contracts.ArtifactRef) map[string]contracts.ArtifactRef {
	result := make(map[string]contracts.ArtifactRef, len(source))
	for name, ref := range source {
		if ref.Revision != nil {
			revision := *ref.Revision
			ref.Revision = &revision
		}
		result[name] = ref
	}
	return result
}

func cloneHTTPOriginTarget(source *contracts.HTTPOriginTargetRef) *contracts.HTTPOriginTargetRef {
	if source == nil {
		return nil
	}
	result := *source
	if source.Credential != nil {
		credential := *source.Credential
		result.Credential = &credential
	}
	return &result
}

func acceptsMediaType(accepted []string, actual string) bool {
	for _, candidate := range accepted {
		if candidate == actual {
			return true
		}
	}
	return false
}
