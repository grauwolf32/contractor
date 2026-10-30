// Package runtimeconfigtest holds the RuntimeConfig fixtures that tests of
// other packages share. Production code never imports it.
package runtimeconfigtest

import (
	"errors"

	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/jackc/pgx/v5"
)

// BuiltInRunSnapshot pins a Run to the built-in RuntimeConfig through the
// default label at binding revision 1 and pins no other label or credential,
// the snapshot a Run without Run labels pins on a freshly migrated database.
func BuiltInRunSnapshot() runtimeconfig.RunSnapshot {
	return runtimeconfig.RunSnapshot{
		Default: runtimeconfig.PinnedLabel{
			Label: runtimeconfig.DefaultLabel, BindingRevision: 1,
			Config: runtimeconfig.Ref{
				Name: runtimeconfig.BuiltInName, Version: runtimeconfig.BuiltInVersion,
				Digest: runtimeconfig.BuiltInDigest,
			},
		},
		Labels: []runtimeconfig.PinnedLabel{}, LLMCredentialIDs: []string{}, RuntimeCredentialIDs: []string{},
	}
}

// PlannerTelemetryAdapterCatalogFunc is a runtimeconfig.PlannerTelemetryAdapterCatalog
// that a test states as a function; production passes the Planner registry.
type PlannerTelemetryAdapterCatalogFunc func(string) bool

func (f PlannerTelemetryAdapterCatalogFunc) SupportsPlannerTelemetryAdapter(ref string) bool {
	return f != nil && f(ref)
}

// TransactionLLMCredentialLookupFactoryFunc is a
// runtimeconfig.TransactionLLMCredentialLookupFactory that a test states as a
// function; production passes the credential store's transaction lookup.
type TransactionLLMCredentialLookupFactoryFunc func(pgx.Tx) (resolved.CredentialLookup, error)

func (f TransactionLLMCredentialLookupFactoryFunc) ForTransaction(tx pgx.Tx) (resolved.CredentialLookup, error) {
	if f == nil {
		return nil, errors.New("transaction LLM credential lookup factory is nil")
	}
	return f(tx)
}
