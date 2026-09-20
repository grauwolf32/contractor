package evalservice

import (
	"context"
	"errors"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditstandards"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/evaldomain"
	pg "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

type Catalog interface {
	ResolveRunWorkflow(context.Context, string, config.ExecutionConfigPatch, config.CredentialLookup) (config.ResolvedWorkflow, error)
	AuditProfile(string) (config.ResolvedAuditProfile, error)
}

type CredentialBarrier interface {
	WithRunCreation(context.Context, func() error) error
}

type Resolver struct {
	Pool        *pgxpool.Pool
	Catalog     Catalog
	Credentials runtimeconfig.TransactionLLMCredentialLookupFactory
	Barrier     CredentialBarrier
}

type BindingSnapshot struct {
	Workflow  *config.ResolvedWorkflow      `json:"workflow,omitempty"`
	Audit     *config.ResolvedAuditProfile  `json:"audit,omitempty"`
	Runtime   runtimeconfig.RunSnapshot     `json:"runtime"`
	Skills    []contracts.RunSkillSnapshot  `json:"skills"`
	Standards []auditstandards.ExactPackage `json:"standards"`
}

// Resolve freezes exact configuration and input metadata in one read transaction.
// It creates no executions or artifacts. Dispatch rechecks the retained pins.
func (r *Resolver) Resolve(ctx context.Context, owner string, v evaldomain.Variant, cases []evaldomain.Case) (Preflight, error) {
	out := Preflight{Pins: map[string]Pin{}, Cases: map[string]Eligibility{}}
	if r.Pool == nil || r.Catalog == nil || r.Credentials == nil || r.Barrier == nil {
		return out, errors.New("eval preflight dependencies are incomplete")
	}
	raw, err := jsonBytes(v)
	if err != nil {
		return out, err
	}
	if err := evaldomain.Validate("Variant", raw); err != nil {
		return out, err
	}
	err = r.Barrier.WithRunCreation(ctx, func() error {
		return pg.InTx(ctx, r.Pool, pgx.TxOptions{IsoLevel: pgx.RepeatableRead}, func(tx pgx.Tx) error {
			tracker, lookup, err := bindPreflightCredentials(tx, r.Credentials)
			if err != nil {
				return err
			}
			service := artifacts.NewService(artifacts.NewPostgresRepository(tx))
			binding, err := r.resolveBinding(ctx, owner, v, tracker, service)
			if err != nil {
				return err
			}
			if err := binding.pinDependencies(ctx, tx, service, owner, v, tracker, lookup); err != nil {
				return err
			}
			out.Capabilities = binding.capabilities()
			for _, c := range cases {
				eligibility, err := binding.evaluateCase(ctx, tx, service, owner, c, v, out.Capabilities)
				if err != nil {
					return err
				}
				out.Cases[c.ID] = eligibility
			}
			for dimension, value := range bindingPinValues(binding.snapshot, binding.workflows) {
				digest, err := hashJSON(value)
				if err != nil {
					return err
				}
				out.Pins[dimension] = observedPin(digest)
			}
			// Catalog model names do not prove provider revisions or runtime builds.
			for _, dimension := range []string{"model-revision", "runtime-build", "tool-docstrings"} {
				out.Pins[dimension] = Pin{Origin: "unavailable"}
			}
			out.Snapshot, err = jsonBytes(binding.snapshot)
			return err
		})
	})
	return out, err
}
