// Package public implements the authenticated single-user HTTP API. Scope IDs
// are derived from authentication and route-owned Run records, never accepted
// as arbitrary request fields.
//
// This file holds the package's dependency surface. Wire bodies live in
// the <family>_types.go file beside their handlers; error bodies live in
// errors.go with the writer that emits them.
package public

import (
	"log/slog"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auth"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/gatewayrecovery"
	publicevents "github.com/grauwolf32/contractor/internal/httpapi/public/events"
)

type Dependencies struct {
	GatewayRecovery        *gatewayrecovery.Service
	Evals                  EvalManagement
	EvalNotifier           interface{ Wake() }
	GitImports             GitImportService
	GitKeys                GitKeySettings
	Authentication         *auth.Service
	BrowserOrigins         auth.OriginPolicy
	InsecureLoopbackCookie bool
	// TrustedPeers resolves the login rate-limit client behind reverse proxies;
	// its zero value attributes every failure to the socket peer.
	TrustedPeers            auth.PeerPolicy
	Config                  ConfigurationCatalog
	ConfigurationPublisher  ConfigurationPublisher
	Credentials             config.CredentialLookup
	ManagedCredentials      ManagedCredentialLifecycle
	RuntimeConfigs          RuntimeConfigManagement
	RuntimeCredentials      RuntimeCredentialManagement
	RuntimeAgentPrincipals  RuntimeAgentPrincipalManagement
	Projects                ProjectManagement
	Audits                  AuditManagement
	FindingProposals        FindingProposalManagement
	FindingCollections      FindingCollectionManagement
	Runs                    RunReader
	RunQueue                RunQueue
	RunLifecycle            RunLifecycle
	RunCreator              RunCreationService
	PlannerPlans            PlannerPlanReader
	Metrics                 MetricsReader
	Operations              OperationsReader
	Performance             PerformanceReader
	AllocationResources     AllocationResourceReader
	OperationsInvalidator   OperationsInvalidator
	SchedulerSettings       SchedulerSettingsManagement
	Events                  *publicevents.Hub
	Artifacts               *artifacts.Service
	BearerToken             contracts.SecretString
	NewID                   func(string) (string, error)
	NewRequestID            func() (string, error)
	RunNotifier             RunNotifier
	ProjectDeletionNotifier RunNotifier
	Now                     func() time.Time
	Logger                  *slog.Logger
}
