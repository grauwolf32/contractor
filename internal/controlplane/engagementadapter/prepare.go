// Package engagementadapter composes authorized engagement inputs into allocation
// preparation. The Control Plane core depends only on its synchronous port.
package engagementadapter

import (
	"context"
	"errors"
	"fmt"
	"time"

	artifactpostgres "github.com/grauwolf32/contractor/internal/artifacts/postgres"
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/control"
	"github.com/grauwolf32/contractor/internal/contracts/engagementcontract"
	"github.com/grauwolf32/contractor/internal/contracts/runtimesettings"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/engagement"
	"github.com/grauwolf32/contractor/internal/runservice"
	"github.com/jackc/pgx/v5/pgxpool"
)

type identityMaterialResolver interface {
	UsePlaintext(context.Context, string, []runtimesettings.RuntimeCredentialKind, func(runtimesettings.RuntimeCredentialKind, []byte) error) error
}

// Adapter owns Run and Audit integration, never Registry state or domain persistence.
// Its lookups finish outside Registry locks and never perform target I/O.
type Adapter struct {
	pool       *pgxpool.Pool
	storage    *artifactpostgres.Storage
	projection func(context.Context, controlplane.Reservation, runtimesettings.RuntimeSettings, bool) (*engagementcontract.AllocationProjection, error)
	materials  func(context.Context, *engagementcontract.AllocationProjection) (map[string]runtimesettings.HTTPIdentityMaterial, error)
}

func New(pool *pgxpool.Pool, identities identityMaterialResolver, storage *artifactpostgres.Storage) (*Adapter, error) {
	if pool == nil || identities == nil || storage == nil {
		return nil, errors.New("engagement adapter dependencies are required")
	}
	return &Adapter{
		pool: pool, storage: storage,
		projection: func(ctx context.Context, reservation controlplane.Reservation, settings runtimesettings.RuntimeSettings, usesHTTP bool) (*engagementcontract.AllocationProjection, error) {
			pin, ordinary, err := runservice.ReadAllocationEngagement(ctx, pool, reservation.Grant.RunID, reservation.Grant.StageExecutionID, storage)
			if err != nil {
				return nil, err
			}
			protected := func(ctx context.Context) ([]engagementcontract.ProtectedDestination, error) {
				return resolveProtectedDestinations(ctx, reservation, settings)
			}
			if !ordinary {
				return auditservice.ResolveAllocationEngagement(ctx, pool, reservation.Grant.RunID, reservation.Grant.StageExecutionID, usesHTTP, protected, storage)
			}
			if pin == nil {
				return nil, nil
			}
			var destinations []engagementcontract.ProtectedDestination
			if usesHTTP {
				destinations, err = protected(ctx)
				if err != nil {
					return nil, err
				}
			}
			return engagement.Project(*pin, usesHTTP, destinations)
		},
		materials: func(ctx context.Context, projection *engagementcontract.AllocationProjection) (map[string]runtimesettings.HTTPIdentityMaterial, error) {
			return engagement.MaterializeHTTPIdentities(ctx, identities, projection)
		},
	}, nil
}

// RequiredRuntimeAdapters derives requirements from selected tools and the immutable Run
// authority before placement, keeping them separate from RuntimeConfig provenance.
func (a *Adapter) RequiredRuntimeAdapters(ctx context.Context, runID, stageID string, bindings []controlplane.BindingRequirement) ([]runtimesettings.RuntimeAdapterRef, error) {
	usesHTTP := false
	for _, binding := range bindings {
		usesHTTP = usesHTTP || resolved.TemplateUsesControlledHTTP(binding.AgentTemplate)
	}
	pin, ordinary, err := runservice.ReadAllocationEngagement(ctx, a.pool, runID, stageID, a.storage)
	if err != nil {
		return nil, err
	}
	if !ordinary {
		return auditservice.AllocationEngagementAdapterRequirements(ctx, a.pool, runID, stageID, usesHTTP, a.storage)
	}
	if pin != nil && usesHTTP {
		return []runtimesettings.RuntimeAdapterRef{runtimesettings.RuntimeAdapterHTTPTarget}, nil
	}
	return nil, nil
}

func (a *Adapter) WithPreparedSpec(ctx context.Context, reservation controlplane.Reservation, spec control.AllocationSpec, dispatch func(control.AllocationSpec) (contracts.WorkerHandle, error)) (contracts.WorkerHandle, error) {
	projection, err := a.projection(ctx, reservation, spec.RuntimeSettings, resolved.TemplateUsesControlledHTTP(spec.AgentTemplate))
	if err != nil {
		return contracts.WorkerHandle{}, err
	}
	spec.Engagement = projection
	uncontrolled := resolved.TemplateRequiresNetwork(spec.AgentTemplate)
	if projection != nil && projection.HTTP != nil {
		uncontrolled = resolved.TemplateRequiresUncontrolledNetwork(spec.AgentTemplate)
	}
	if projection != nil && uncontrolled {
		return contracts.WorkerHandle{}, fmt.Errorf("%w: engagement metadata does not authorize target transport", controlplane.ErrInvalidRequest)
	}
	if projection != nil && projection.HTTP != nil {
		materialCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
		material, materialErr := a.materials(materialCtx, projection)
		cancel()
		defer clearMaterials(material)
		if materialErr != nil {
			return contracts.WorkerHandle{}, materialErr
		}
		spec.RuntimeSettings.HTTPIdentities = material
	}
	return dispatch(spec)
}

func clearMaterials(material map[string]runtimesettings.HTTPIdentityMaterial) {
	for id, value := range material {
		clear(value.Fields)
		delete(material, id)
	}
}

var _ controlplane.AllocationSpecPreparation = (*Adapter)(nil)
