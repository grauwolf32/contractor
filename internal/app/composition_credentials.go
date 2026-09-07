package app

import (
	"context"
	"errors"
	"fmt"
	"math"

	"github.com/grauwolf32/contractor/internal/auditstore"
	workflowconfig "github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/credentials"
	litellmcredentials "github.com/grauwolf32/contractor/internal/credentials/litellm"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/jackc/pgx/v5/pgxpool"
)

// credentialServices holds the providers and lifecycle services shared by Server components.
type credentialServices struct {
	provider          *credentials.CompositeProvider
	lifecycle         *credentials.Service
	runtime           *credentials.RuntimeCredentialService
	transactionLookup *credentials.TransactionLookupFactory
	gitKeys           *credentials.GitKeys
}

func configureCredentials(
	ctx context.Context,
	pool *pgxpool.Pool,
	configurationManager *workflowconfig.Manager,
	cfg Config,
) (credentialServices, error) {
	snapshot := configurationManager.Snapshot()
	credentialRepository := credentials.NewRepository(pool)
	activeCredentialCount, err := credentialRepository.CountCredentials(ctx)
	if err != nil {
		return credentialServices{}, fmt.Errorf("inspect encrypted LLM credentials: %w", err)
	}
	runtimeCredentialRepository := credentials.NewRuntimeCredentialRepository(pool)
	storedRuntimeCredentialCount, err := runtimeCredentialRepository.CountStored(ctx)
	if err != nil {
		return credentialServices{}, fmt.Errorf("inspect encrypted Runtime credentials: %w", err)
	}
	if storedRuntimeCredentialCount > 0 && activeCredentialCount > math.MaxInt64-storedRuntimeCredentialCount {
		return credentialServices{}, errors.New("encrypted credential count is invalid")
	}
	activeCredentialCount += storedRuntimeCredentialCount
	gitKeyCount, err := credentials.NewGitKeys(pool, nil).Count(ctx)
	if err != nil {
		return credentialServices{}, errors.New("inspect encrypted Git keys")
	}
	if gitKeyCount > math.MaxInt64-activeCredentialCount {
		return credentialServices{}, errors.New("encrypted credential count is invalid")
	}
	activeCredentialCount += gitKeyCount
	tokenCipher, err := credentials.RequireTokenCipher(cfg.CredentialMasterKeyFile, activeCredentialCount)
	if err != nil {
		return credentialServices{}, fmt.Errorf("configure encrypted LLM credentials: %w", err)
	}
	if tokenCipher != nil {
		if err := credentialRepository.VerifyActiveKey(ctx, tokenCipher.KeyID()); err != nil {
			return credentialServices{}, fmt.Errorf("verify encrypted LLM credential key: %w", err)
		}
		if err := runtimeCredentialRepository.VerifyStoredKey(ctx, tokenCipher.KeyID()); err != nil {
			return credentialServices{}, fmt.Errorf("verify encrypted Runtime credential key: %w", err)
		}
	}
	gitKeys := credentials.NewGitKeys(pool, tokenCipher)
	if err := gitKeys.Verify(ctx); err != nil {
		return credentialServices{}, errors.New("verify encrypted Git keys")
	}
	encryptedCredentialProvider, err := credentials.NewEncryptedProvider(credentialRepository, tokenCipher)
	if err != nil {
		return credentialServices{}, fmt.Errorf("configure encrypted LLM credentials: %w", err)
	}
	developmentCredentialProvider, err := developmentCredentials(snapshot, cfg)
	if err != nil {
		return credentialServices{}, fmt.Errorf("configure development LLM credentials: %w", err)
	}
	transactionCredentialLookup, err := credentials.NewTransactionLookupFactory(developmentCredentialProvider)
	if err != nil {
		return credentialServices{}, fmt.Errorf("configure transaction-bound LLM credential lookup: %w", err)
	}
	credentialProvider, err := credentials.NewCompositeProvider(
		developmentCredentialProvider, encryptedCredentialProvider,
	)
	if err != nil {
		return credentialServices{}, fmt.Errorf("compose LLM credential providers: %w", err)
	}
	adminBindings, err := litellmcredentials.LoadAdminBindings(
		cfg.LLMGatewayAdminBindingsFile, configurationManager,
	)
	if err != nil {
		return credentialServices{}, fmt.Errorf("load LLM Gateway admin bindings: %w", err)
	}
	if adminBindings.Len() != 0 && tokenCipher == nil {
		return credentialServices{}, errors.New("managed LLM Gateway credentials require --credential-master-key-file")
	}
	liteLLMManager, err := litellmcredentials.NewManager(
		adminBindings, configurationManager, litellmcredentials.Options{},
	)
	if err != nil {
		return credentialServices{}, fmt.Errorf("configure LiteLLM credential manager: %w", err)
	}
	credentialManagers, err := credentials.NewManagerRegistry(credentials.ManagerRegistration{
		Implementation: contracts.LiteLLMVirtualKeysManager,
		Manager:        liteLLMManager,
	})
	if err != nil {
		return credentialServices{}, fmt.Errorf("configure Gateway credential managers: %w", err)
	}
	credentialBarrier := credentials.NewLifecycleBarrier()
	credentialLifecycle, err := credentials.NewService(credentials.ServiceOptions{
		Pool: pool, Gateways: configurationManager, Managers: credentialManagers,
		Runs: runstore.NewPostgresStore(pool), Audits: auditstore.NewPostgresStore(pool),
		Cipher: tokenCipher, Barrier: credentialBarrier,
	})
	if err != nil {
		return credentialServices{}, fmt.Errorf("configure LLM credential lifecycle: %w", err)
	}
	if err := credentialLifecycle.Recover(ctx); err != nil {
		return credentialServices{}, fmt.Errorf("recover LLM credential operations: %w", err)
	}
	runtimeCredentialLifecycle, err := credentials.NewRuntimeCredentialService(
		credentials.RuntimeCredentialServiceOptions{
			Pool: pool, Cipher: tokenCipher, Usage: runtimeCredentialRepository,
			Barrier: credentialBarrier,
		},
	)
	if err != nil {
		return credentialServices{}, fmt.Errorf("configure Runtime credential lifecycle: %w", err)
	}
	return credentialServices{
		provider: credentialProvider, lifecycle: credentialLifecycle,
		runtime: runtimeCredentialLifecycle, transactionLookup: transactionCredentialLookup,
		gitKeys: gitKeys,
	}, nil
}

const (
	developmentWorkerCredential  = "development-worker"
	developmentPlannerCredential = "development-planner"
)

func developmentCredentials(
	snapshot *workflowconfig.Snapshot,
	cfg Config,
) (*credentials.StaticProvider, error) {
	entries := make([]credentials.StaticEntry, 0, 2)
	if cfg.DevelopmentWorkerToken.Reveal() == "" && cfg.DevelopmentPlannerToken.Reveal() == "" {
		return credentials.NewStaticProvider(entries)
	}
	gateway, err := snapshot.LLMGateway("local-litellm@1")
	if err != nil {
		return nil, errors.New("development tokens require LLMGatewayConfig local-litellm@1")
	}
	appendEntry := func(id string, token contracts.SecretString) {
		if token.Reveal() == "" {
			return
		}
		entries = append(entries, credentials.StaticEntry{
			Metadata: workflowconfig.CredentialMetadata{
				Ref: contracts.LLMCredentialRef{CredentialID: id}, LLMGateway: gateway.Ref,
				Unrestricted: true,
			},
			Token: token,
		})
	}
	appendEntry(developmentWorkerCredential, cfg.DevelopmentWorkerToken)
	appendEntry(developmentPlannerCredential, cfg.DevelopmentPlannerToken)
	return credentials.NewStaticProvider(entries)
}
