package runservice

import "fmt"

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
