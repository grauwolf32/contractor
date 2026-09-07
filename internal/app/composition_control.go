package app

import (
	"context"
	"crypto/tls"
	"fmt"

	workflowconfig "github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/mtls"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/grauwolf32/contractor/internal/telemetry"
	"github.com/jackc/pgx/v5/pgxpool"
)

type controlServices struct {
	tls                   *tls.Config
	registry              *controlplane.InMemoryRegistry
	principals            *runtimeconfig.PrincipalService
	principalOperations   *controlplane.PrincipalOperations
	allocator             *controlplane.PlacementAllocator
	runtimeClient         *controlplane.RuntimeControlClient
	workers               *controlplane.RuntimeBatchController
	runtimeConfigurations *runtimeconfig.ManagementService
}

func configureControlPlane(
	pool *pgxpool.Pool,
	configurationManager *workflowconfig.Manager,
	cfg Config,
	credentialSet credentialServices,
	plannerTelemetryRegistry *telemetry.PlannerAdapterRegistry,
) (controlServices, error) {
	runtimeConfigPublisher, err := runtimeconfig.NewPublisher(runtimeconfig.PublisherOptions{
		Pool: pool,
		GatewayResolver: runtimeconfig.GatewayResolverFunc(func(_ context.Context, selector string) (contracts.ResolvedLLMGatewayConfig, error) {
			return configurationManager.LLMGateway(selector)
		}),
		RuntimeCredentials:       credentialSet.runtime,
		PlannerTelemetryAdapters: plannerTelemetryRegistry,
	})
	if err != nil {
		return controlServices{}, fmt.Errorf("configure RuntimeConfig publisher: %w", err)
	}
	runtimeBindingService, err := runtimeconfig.NewBindingService(pool, credentialSet.runtime)
	if err != nil {
		return controlServices{}, fmt.Errorf("configure Runtime label bindings: %w", err)
	}
	runtimeConfigManagement, err := runtimeconfig.NewManagementService(
		pool, runtimeConfigPublisher, runtimeBindingService,
	)
	if err != nil {
		return controlServices{}, fmt.Errorf("configure RuntimeConfig management: %w", err)
	}
	files := mtls.Files{
		Certificate: cfg.CertificateFile, PrivateKey: cfg.PrivateKeyFile, CA: cfg.CAFile,
	}
	privateTLS, err := mtls.ControlPlaneServerConfig(files)
	if err != nil {
		return controlServices{}, fmt.Errorf("configure private mTLS server: %w", err)
	}
	registry, err := controlplane.NewRegistry(controlplane.RegistryOptions{})
	if err != nil {
		return controlServices{}, fmt.Errorf("configure Control Plane registry: %w", err)
	}
	principalService, err := runtimeconfig.NewPrincipalService(runtimeconfig.PrincipalServiceOptions{
		Pool: pool, DeletionGuard: registry,
	})
	if err != nil {
		return controlServices{}, fmt.Errorf("configure Runtime Agent principals: %w", err)
	}
	principalOperations, err := controlplane.NewPrincipalOperations(principalService, registry)
	if err != nil {
		return controlServices{}, fmt.Errorf("configure Runtime Agent principal Operations: %w", err)
	}
	placementAllocator, err := controlplane.NewPlacementAllocator(controlplane.PlacementAllocatorOptions{
		Pool: pool, Registry: registry, Gateways: configurationManager,
		LLMCredentials: credentialSet.provider, RuntimeCredentials: credentialSet.runtime,
		CredentialGuard: credentialSet.lifecycle, PerformanceMetrics: cfg.PerformanceMetrics,
	})
	if err != nil {
		return controlServices{}, fmt.Errorf("configure candidate Runtime placement: %w", err)
	}
	runtimeClient, err := controlplane.NewMTLSRuntimeControlClient(files, cfg.RuntimeRequestTimeout)
	if err != nil {
		return controlServices{}, fmt.Errorf("configure Runtime Agent client: %w", err)
	}
	workers, err := controlplane.NewRuntimeBatchController(
		runtimeClient,
		registry,
		controlplane.RuntimeBatchOptions{CleanupTimeout: cfg.Operations.RuntimeLifecycle.CleanupTimeout},
	)
	if err != nil {
		return controlServices{}, fmt.Errorf("configure Runtime Agent lifecycle: %w", err)
	}
	return controlServices{
		tls: privateTLS, registry: registry, principals: principalService,
		principalOperations: principalOperations, allocator: placementAllocator,
		runtimeClient: runtimeClient, workers: workers,
		runtimeConfigurations: runtimeConfigManagement,
	}, nil
}
