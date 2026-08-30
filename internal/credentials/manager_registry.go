package credentials

import (
	"fmt"

	"github.com/grauwolf32/contractor/internal/contracts"
)

// ManagerRegistration binds one immutable implementation identifier to its
// process-local outbound adapter. Gateway-specific admin bindings remain owned
// by the adapter and are selected using the complete digest-bearing ref.
type ManagerRegistration struct {
	Implementation string
	Manager        GatewayCredentialManager
}

type ManagerRegistry struct {
	managers map[string]GatewayCredentialManager
}

func NewManagerRegistry(registrations ...ManagerRegistration) (*ManagerRegistry, error) {
	result := &ManagerRegistry{managers: make(map[string]GatewayCredentialManager, len(registrations))}
	for _, registration := range registrations {
		if registration.Implementation == "" || registration.Manager == nil {
			return nil, fmt.Errorf("%w: invalid manager registration", ErrInvalid)
		}
		if _, exists := result.managers[registration.Implementation]; exists {
			return nil, fmt.Errorf("%w: duplicate manager implementation", ErrConflict)
		}
		result.managers[registration.Implementation] = registration.Manager
	}
	return result, nil
}

func (r *ManagerRegistry) ForGateway(
	gateway contracts.ResolvedLLMGatewayConfig,
) (GatewayCredentialManager, error) {
	if r == nil || gateway.Validate() != nil || gateway.CredentialManager == nil {
		return nil, ErrManagerUnavailable
	}
	manager, exists := r.managers[gateway.CredentialManager.Implementation]
	if !exists || manager == nil {
		return nil, ErrManagerUnavailable
	}
	return manager, nil
}
