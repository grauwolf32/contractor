package auditservice

import (
	"context"
	"fmt"

	artifactpostgres "github.com/grauwolf32/contractor/internal/artifacts/postgres"
	"github.com/grauwolf32/contractor/internal/auditbaseline"
	"github.com/grauwolf32/contractor/internal/auditstore/querystore"
	"github.com/grauwolf32/contractor/internal/contracts/engagementcontract"
	"github.com/grauwolf32/contractor/internal/contracts/runtimesettings"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// allocationEngagement reads only the immutable Run binding and Audit baseline.
// It finishes the snapshot before any DNS or private HTTP operation.
func allocationEngagement(ctx context.Context, pool *pgxpool.Pool, runID, stageID string, storage *artifactpostgres.Storage) (string, *auditbaseline.EngagementSnapshot, error) {
	var auditID string
	var pin *auditbaseline.EngagementSnapshot
	err := persistencepostgres.InTx(ctx, pool, pgx.TxOptions{AccessMode: pgx.ReadOnly, IsoLevel: pgx.RepeatableRead}, func(tx pgx.Tx) error {
		binding, err := runstore.NewPostgresStore(tx, storage).ReadAllocationAuditBinding(ctx, runID, stageID)
		if err != nil {
			return err
		}
		if binding.ExecutionID == nil {
			return nil
		}
		id, encoded, err := querystore.New(tx, storage).ReadAllocationBaseline(ctx, binding.OwnerID, *binding.ProjectID, *binding.ExecutionID, runID)
		if err != nil {
			return err
		}
		baseline, err := DecodeBaseline(encoded)
		if err != nil {
			return err
		}
		if baseline.Engagement == nil {
			return nil
		}
		if baseline.Engagement.AuthorizedBy != binding.OwnerID {
			return fmt.Errorf("stored engagement authorizer differs from Run owner")
		}
		auditID, pin = id, baseline.Engagement
		return nil
	})
	return auditID, pin, err
}

// AllocationEngagementAdapterRequirements adds only tool-selected controlled
// execution to placement. Ordinary and metadata-only allocations add nothing.
func AllocationEngagementAdapterRequirements(ctx context.Context, pool *pgxpool.Pool, runID, stageID string, usesHTTP bool, storage *artifactpostgres.Storage) ([]runtimesettings.RuntimeAdapterRef, error) {
	_, pin, err := allocationEngagement(ctx, pool, runID, stageID, storage)
	if err != nil {
		return nil, err
	}
	if pin != nil && usesHTTP {
		return []runtimesettings.RuntimeAdapterRef{runtimesettings.RuntimeAdapterHTTPTarget}, nil
	}
	return nil, nil
}

// ResolveAllocationEngagement derives the narrow private assignment from the
// immutable Audit binding. Protected endpoint resolution runs after its two
// bounded database reads, outside the transaction and Registry mutex.
func ResolveAllocationEngagement(ctx context.Context, pool *pgxpool.Pool, runID, stageID string, usesHTTP bool, protected func(context.Context) ([]engagementcontract.ProtectedDestination, error), storage *artifactpostgres.Storage) (*engagementcontract.AllocationProjection, error) {
	auditID, pin, err := allocationEngagement(ctx, pool, runID, stageID, storage)
	if err != nil || pin == nil {
		return nil, err
	}
	// pin is freshly decoded for this call; transfer its fields to the result.
	result := &engagementcontract.AllocationProjection{
		Schema: "contractor.engagement.allocation.v1", AuditID: auditID,
		Engagement: pin.Artifact.Ref, EngagementDigest: pin.Artifact.Digest,
		AuthorizedBy: pin.AuthorizedBy, AuthorizedAt: pin.AuthorizedAt,
	}
	if usesHTTP {
		if protected == nil {
			return nil, fmt.Errorf("controlled preflight requires trusted infrastructure destinations")
		}
		destinations, err := protected(ctx)
		if err != nil {
			return nil, err
		}
		assignment := &engagementcontract.HTTPAssignment{
			Adapter:               runtimesettings.RuntimeAdapterHTTPTarget,
			Routes:                pin.Document.Egress.Routes,
			Exclusions:            pin.Document.Egress.Exclusions,
			ProtectedDestinations: destinations,
			Identities:            pin.Document.Identities,
		}
		assignment.Limits = pin.Document.Limits
		result.HTTP = assignment
	}
	if err := result.Validate(); err != nil {
		return nil, fmt.Errorf("stored engagement allocation projection is invalid: %w", err)
	}
	return result, nil
}
