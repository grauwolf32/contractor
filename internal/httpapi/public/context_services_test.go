package public

import (
	"context"
	"errors"
	"log/slog"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auth"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/credentials"
	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/grauwolf32/contractor/internal/evalservice"
	"github.com/grauwolf32/contractor/internal/evalstore"
	"github.com/grauwolf32/contractor/internal/findingintake"
	"github.com/grauwolf32/contractor/internal/gatewayrecovery"
	"github.com/grauwolf32/contractor/internal/gitimport"
	publicevents "github.com/grauwolf32/contractor/internal/httpapi/public/events"
)

// testCatalog is the published configuration a test serves: what the
// catalog, Run repeat drafts and Evals read, and what the test Run creator
// resolves Workflows with.
type testCatalog interface {
	CatalogReader
	RepeatWorkflowResolver
	WorkflowLister
}

// testManagedCredentials is the managed LLM credential service a test serves;
// the test Run creator also takes its Run creation guard.
type testManagedCredentials interface {
	ManagedCredentialLifecycle
	WithRunCreation(context.Context, func() error) error
}

// testRuntimeCredentials serves the Runtime credential routes and the
// credential checks of Project targets and Run creation.
type testRuntimeCredentials interface {
	RuntimeCredentialManagement
	ProjectCredentialValidator
}

// testServices is the flat set of services a test handler is built from.
// dependencies groups them by context the way the Server's composition does,
// so a test replaces one service for every context that uses it.
type testServices struct {
	BearerToken  contracts.SecretString
	NewRequestID func() (string, error)
	Logger       *slog.Logger
	NewID        func(string) (string, error)
	Now          func() time.Time

	Authentication         *auth.Service
	BrowserOrigins         auth.OriginPolicy
	TrustedPeers           auth.PeerPolicy
	InsecureLoopbackCookie bool
	Events                 *publicevents.Hub

	Config                 testCatalog
	ConfigurationPublisher ConfigurationPublisher
	Credentials            config.CredentialLookup
	ManagedCredentials     testManagedCredentials
	RuntimeConfigs         RuntimeConfigManagement
	RuntimeCredentials     testRuntimeCredentials
	RuntimeAgentPrincipals RuntimeAgentPrincipalManagement

	Projects                ProjectManagement
	ProjectDeletionNotifier Waker
	Runs                    RunReader
	RunDetails              RunDetailReader
	RunQueue                RunQueue
	RunLifecycle            RunLifecycle
	RunDeletion             RunDeletion
	RunCreator              RunCreationService
	RunNotifier             RunNotifier
	GatewayRecovery         GatewayRecovery
	Artifacts               *artifacts.Service
	GitImports              GitImportService
	GitKeys                 GitKeySettings

	Audits             AuditManagement
	FindingProposals   FindingProposalManagement
	FindingCollections FindingCollectionManagement
	Evals              EvalManagement
	EvalNotifier       Waker

	Operations            OperationsReader
	OperationsInvalidator OperationsInvalidator
	Performance           PerformanceReader
	AllocationResources   AllocationResourceReader
	SchedulerSettings     SchedulerSettingsManagement
}

func (s testServices) dependencies() Dependencies {
	return Dependencies{
		BearerToken: s.BearerToken, NewRequestID: s.NewRequestID, Logger: s.Logger,
		Session: SessionDependencies{
			Authentication: s.Authentication, BrowserOrigins: s.BrowserOrigins,
			TrustedPeers: s.TrustedPeers, InsecureLoopbackCookie: s.InsecureLoopbackCookie,
			Events: s.Events,
		},
		Catalog: CatalogDependencies{
			Config: s.Config, ConfigurationPublisher: s.ConfigurationPublisher,
			OperationsInvalidator: s.OperationsInvalidator,
		},
		Projects: ProjectDependencies{
			Projects: s.Projects, RuntimeCredentials: s.RuntimeCredentials,
			ProjectDeletionNotifier: s.ProjectDeletionNotifier, NewID: s.NewID,
		},
		Runs: RunDependencies{
			Runs: s.Runs, RunDetails: s.RunDetails, RunLifecycle: s.RunLifecycle,
			RunDeletion: s.RunDeletion, RunCreator: s.RunCreator, RunNotifier: s.RunNotifier,
			GatewayRecovery: s.GatewayRecovery, Projects: s.Projects, NewID: s.NewID, Now: s.Now,
		},
		RunRepeat: RunRepeatDependencies{
			Runs: s.Runs, Projects: s.Projects, Config: s.Config, Credentials: s.Credentials,
			Artifacts: s.Artifacts, RuntimeConfigs: s.RuntimeConfigs,
		},
		Queue: QueueDependencies{RunQueue: s.RunQueue, RunNotifier: s.RunNotifier},
		Artifacts: ArtifactDependencies{
			Artifacts: s.Artifacts, Projects: s.Projects, Runs: s.Runs, GitImports: s.GitImports,
		},
		Audits: AuditDependencies{Audits: s.Audits, Projects: s.Projects, NewID: s.NewID},
		Findings: FindingDependencies{
			FindingProposals: s.FindingProposals, FindingCollections: s.FindingCollections,
			Audits: s.Audits, Runs: s.Runs,
		},
		Evals: EvalDependencies{
			Evals: s.Evals, EvalNotifier: s.EvalNotifier, Config: s.Config, Audits: s.Audits,
		},
		Operations: OperationsDependencies{
			Operations: s.Operations, Performance: s.Performance,
			AllocationResources: s.AllocationResources, SchedulerSettings: s.SchedulerSettings,
			OperationsInvalidator: s.OperationsInvalidator, RunNotifier: s.RunNotifier, Logger: s.Logger,
		},
		Credentials: CredentialDependencies{
			ManagedCredentials: s.ManagedCredentials, OperationsInvalidator: s.OperationsInvalidator,
		},
		RuntimeConfiguration: RuntimeConfigurationDependencies{
			RuntimeConfigs: s.RuntimeConfigs, RuntimeCredentials: s.RuntimeCredentials,
			RuntimeAgentPrincipals: s.RuntimeAgentPrincipals,
			OperationsInvalidator:  s.OperationsInvalidator, Logger: s.Logger, Now: s.Now,
		},
		GitKeys: GitKeyDependencies{GitKeys: s.GitKeys},
	}
}

// unavailableGitImports answers like a Server whose Git import is not
// configured.
type unavailableGitImports struct{}

func (unavailableGitImports) DoImport(context.Context, gitimport.ImportRequest, func(gitimport.ImportResult)) error {
	return gitimport.ErrConfiguration
}

// unavailableGitKeys answers like a Server without a Git key encryption key.
type unavailableGitKeys struct{}

func (unavailableGitKeys) Metadata(context.Context, string) (credentials.GitKeyMetadata, error) {
	return credentials.GitKeyMetadata{}, credentials.ErrGitKeyUnavailable
}

func (unavailableGitKeys) Replace(context.Context, string, []byte) (credentials.GitKeyMetadata, error) {
	return credentials.GitKeyMetadata{}, credentials.ErrGitKeyUnavailable
}

func (unavailableGitKeys) Delete(context.Context, string) error {
	return credentials.ErrGitKeyUnavailable
}

// missingFindingProposals holds no finding proposal of any Run or Audit.
type missingFindingProposals struct{}

func (missingFindingProposals) ListRun(context.Context, string, string, findingintake.ListQuery) ([]findingintake.Receipt, error) {
	return nil, findingintake.ErrNotFound
}

func (missingFindingProposals) ListAuditInbox(context.Context, string, string, findingintake.ListQuery) ([]findingintake.Receipt, error) {
	return nil, findingintake.ErrNotFound
}

func (missingFindingProposals) ImportIntoAudit(context.Context, findingintake.ImportRequest) (findingintake.AuditHold, bool, error) {
	return findingintake.AuditHold{}, false, findingintake.ErrNotFound
}

// missingFindingCollections publishes no finding collection.
type missingFindingCollections struct{}

func (missingFindingCollections) PublishCollection(context.Context, findingintake.PublishCollectionParams) (findingintake.PublishedCollection, error) {
	return findingintake.PublishedCollection{}, findingintake.ErrNotFound
}

// noGatewayWait has no Run waiting for its gateway.
type noGatewayWait struct{}

func (noGatewayWait) Retry(context.Context, string, string) error {
	return gatewayrecovery.ErrUnavailable
}

// errEvalsNotServed is the answer of a fixture that serves no managed Evals;
// the Eval tests build their handler on the PostgreSQL Eval service.
var errEvalsNotServed = errors.New("managed Evals are not served by this test fixture")

type unservedEvals struct{}

func (unservedEvals) Get(context.Context, string, string) (evalservice.ExperimentView, error) {
	return evalservice.ExperimentView{}, errEvalsNotServed
}

func (unservedEvals) List(context.Context, evalstore.SummaryPageParams) (evalstore.SummaryPage, error) {
	return evalstore.SummaryPage{}, errEvalsNotServed
}

func (unservedEvals) Datasets(context.Context, evalstore.Scope, string, string, int, *int64) (evalstore.DatasetPage, error) {
	return evalstore.DatasetPage{}, errEvalsNotServed
}

func (unservedEvals) Dataset(context.Context, evalstore.Scope, string, string) (evalstore.DatasetRevision, error) {
	return evalstore.DatasetRevision{}, errEvalsNotServed
}

func (unservedEvals) PutDataset(context.Context, evalstore.Scope, evaldomain.Frozen, evaldomain.MutationIdentity) (evalstore.Receipt, error) {
	return evalstore.Receipt{}, errEvalsNotServed
}

func (unservedEvals) Create(context.Context, evalstore.Scope, evaldomain.Frozen, evaldomain.MutationIdentity) (evalstore.Receipt, error) {
	return evalstore.Receipt{}, errEvalsNotServed
}

func (unservedEvals) ScopeForMutation(context.Context, string, string, string, string) (evalstore.Scope, error) {
	return evalstore.Scope{}, errEvalsNotServed
}

func (unservedEvals) UpdateDraft(context.Context, evalstore.Scope, string, evaldomain.Frozen, evaldomain.MutationIdentity) (evalstore.Receipt, error) {
	return evalstore.Receipt{}, errEvalsNotServed
}

func (unservedEvals) Command(context.Context, evalstore.Scope, string, evaldomain.Command, evaldomain.MutationIdentity) (evalstore.Receipt, error) {
	return evalstore.Receipt{}, errEvalsNotServed
}

func (unservedEvals) Submit(context.Context, evalstore.Scope, string, string, evaldomain.Submission, evaldomain.MutationIdentity) (evalstore.Receipt, error) {
	return evalstore.Receipt{}, errEvalsNotServed
}

func (unservedEvals) Delete(context.Context, evalstore.Scope, string, evaldomain.MutationIdentity) (evalstore.Receipt, error) {
	return evalstore.Receipt{}, errEvalsNotServed
}

func (unservedEvals) GetCommand(context.Context, string, string, string) (evalstore.CommandRecord, *string, error) {
	return evalstore.CommandRecord{}, nil, errEvalsNotServed
}

func (unservedEvals) Members(context.Context, evalservice.MemberPageParams) (evalservice.MemberPage, error) {
	return evalservice.MemberPage{}, errEvalsNotServed
}

func (unservedEvals) PutRecord(context.Context, evalstore.Scope, string, string, evaldomain.Frozen, evaldomain.MutationIdentity) (evalstore.Receipt, error) {
	return evalstore.Receipt{}, errEvalsNotServed
}

func (unservedEvals) Select(context.Context, evalstore.Scope, string, evaldomain.SelectionInput, evaldomain.MutationIdentity) (evalstore.Receipt, error) {
	return evalstore.Receipt{}, errEvalsNotServed
}

func (unservedEvals) Review(context.Context, string, string, string, string) (evalservice.ReviewContext, error) {
	return evalservice.ReviewContext{}, errEvalsNotServed
}

func (unservedEvals) Pairs(context.Context, evalservice.PairPageParams) (evalservice.PairPage, error) {
	return evalservice.PairPage{}, errEvalsNotServed
}

func (unservedEvals) Pair(context.Context, evalservice.MemberPageParams, string) (evalservice.PairDetail, error) {
	return evalservice.PairDetail{}, errEvalsNotServed
}

func (unservedEvals) Chart(context.Context, evalservice.ChartParams) (evalservice.ChartView, error) {
	return evalservice.ChartView{}, errEvalsNotServed
}

func (unservedEvals) Report(context.Context, evalservice.MemberPageParams) (evalservice.Report, error) {
	return evalservice.Report{}, errEvalsNotServed
}

func (unservedEvals) Executions(context.Context, string, string, string, string, int, *int64) (evalstore.InventoryPage, error) {
	return evalstore.InventoryPage{}, errEvalsNotServed
}

// countingWaker counts the wake-ups of a background controller.
type countingWaker struct{ calls int }

func (w *countingWaker) Wake() { w.calls++ }
