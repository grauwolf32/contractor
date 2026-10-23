package app

import (
	"context"
	"crypto/tls"
	"fmt"
	"log/slog"
	"time"

	"github.com/grauwolf32/contractor/internal/agentprincipal"
	workflowconfig "github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts/llmgateway"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/mtls"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/grauwolf32/contractor/internal/telemetry"
	"github.com/jackc/pgx/v5/pgxpool"
)

type controlServices struct {
	tls                   *tls.Config
	registry              *controlplane.InMemoryRegistry
	principals            *agentprincipal.Service
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
	logger *slog.Logger,
) (controlServices, error) {
	runtimeConfigPublisher, err := runtimeconfig.NewPublisher(runtimeconfig.PublisherOptions{
		Pool: pool,
		GatewayResolver: runtimeconfig.GatewayResolverFunc(func(_ context.Context, selector string) (llmgateway.ResolvedLLMGatewayConfig, error) {
			return configurationManager.LLMGateway(selector)
		}),
		RuntimeCredentials:        credentialSet.runtime,
		TransactionLLMCredentials: credentialSet.transactionLookup,
		PlannerTelemetryAdapters:  plannerTelemetryRegistry,
	})
	if err != nil {
		return controlServices{}, fmt.Errorf("configure RuntimeConfig publisher: %w", err)
	}
	runtimeBindingService, err := runtimeconfig.NewBindingService(credentialSet.runtime, credentialSet.transactionLookup)
	if err != nil {
		return controlServices{}, fmt.Errorf("configure Runtime label bindings: %w", err)
	}
	runtimeConfigManagement, err := runtimeconfig.NewManagementService(
		pool, runtimeConfigPublisher, runtimeBindingService, agentprincipal.LabelHolders{},
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
	if err := logControlPlaneCertificateExpiry(logger, cfg.CertificateFile, cfg.CAFile,
		cfg.CertificateExpiryWarningWindow, time.Now); err != nil {
		return controlServices{}, err
	}
	// A waiting model Stage retries placement once per Scheduler poll while a
	// model lane is free; five polls keep its hold across slow attempts and
	// let it lapse soon after the Stage stops waiting (CTL-R155).
	registry, err := controlplane.NewRegistry(controlplane.RegistryOptions{
		ModelStageHold: 5 * cfg.Operations.Scheduler.PollInterval,
	})
	if err != nil {
		return controlServices{}, fmt.Errorf("configure Control Plane registry: %w", err)
	}
	principalService, err := agentprincipal.NewService(agentprincipal.ServiceOptions{
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
		TransactionLLMCredentials: credentialSet.transactionLookup,
		CredentialGuard:           credentialSet.lifecycle, PerformanceMetrics: cfg.PerformanceMetrics,
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
		controlplane.RuntimeBatchOptions{
			PrepareTimeout: cfg.Operations.RuntimeLifecycle.PrepareTimeout,
			CleanupTimeout: cfg.Operations.RuntimeLifecycle.CleanupTimeout,
		},
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
