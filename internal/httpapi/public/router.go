package public

import (
	"crypto/sha256"
	"fmt"
	"log/slog"
	"net/http"
	"reflect"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/requestid"
)

// Dependencies groups the services of the public API by the context that
// uses them. NewHandler rejects a missing service; a service that several
// contexts use is passed to each of them.
type Dependencies struct {
	// BearerToken authenticates non-browser clients and keys the HMAC of
	// every continuation cursor.
	BearerToken  contracts.SecretString
	NewRequestID func() (string, error)
	Logger       *slog.Logger

	Session              SessionDependencies
	Catalog              CatalogDependencies
	Projects             ProjectDependencies
	Runs                 RunDependencies
	RunRepeat            RunRepeatDependencies
	Queue                QueueDependencies
	Artifacts            ArtifactDependencies
	Audits               AuditDependencies
	Findings             FindingDependencies
	Evals                EvalDependencies
	Operations           OperationsDependencies
	Credentials          CredentialDependencies
	RuntimeConfiguration RuntimeConfigurationDependencies
	GitKeys              GitKeyDependencies
}

// contextHandlers holds the handler struct of every context. Only NewHandler
// and the route registration see more than one context.
type contextHandlers struct {
	session              *sessionHandlers
	catalog              *catalogHandlers
	projects             *projectHandlers
	runs                 *runHandlers
	runRepeat            *runRepeatHandlers
	queue                *queueHandlers
	artifacts            *artifactHandlers
	audits               *auditHandlers
	findings             *findingHandlers
	evals                *evalHandlers
	operations           *operationsHandlers
	credentials          *credentialHandlers
	runtimeConfiguration *runtimeConfigurationHandlers
	gitKeys              *gitKeyHandlers
}

func NewHandler(dependencies Dependencies) (http.Handler, error) {
	if dependencies.NewRequestID == nil || dependencies.Logger == nil {
		return nil, fmt.Errorf("public API request identity dependencies are missing")
	}
	for _, group := range []struct {
		name         string
		dependencies any
	}{
		{"session", dependencies.Session},
		{"catalog", dependencies.Catalog},
		{"Project", dependencies.Projects},
		{"Run", dependencies.Runs},
		{"Run repeat", dependencies.RunRepeat},
		{"Run Queue", dependencies.Queue},
		{"Artifact", dependencies.Artifacts},
		{"Audit", dependencies.Audits},
		{"finding", dependencies.Findings},
		{"Eval", dependencies.Evals},
		{"Operations", dependencies.Operations},
		{"credential", dependencies.Credentials},
		{"Runtime configuration", dependencies.RuntimeConfiguration},
		{"Git key", dependencies.GitKeys},
	} {
		if err := requireServices(group.name, group.dependencies); err != nil {
			return nil, err
		}
	}
	if len(dependencies.Session.BrowserOrigins.Values()) == 0 {
		return nil, fmt.Errorf("public API session dependency BrowserOrigins is empty")
	}
	bearerToken := dependencies.BearerToken.Reveal()
	if bearerToken == "" || len(bearerToken) > 4096 {
		return nil, fmt.Errorf("public API bearer token must contain 1 through 4096 bytes")
	}
	tokenDigest := sha256.Sum256([]byte(bearerToken))
	bearerToken = ""
	cursors := pageCursors{key: tokenDigest}
	session := &sessionHandlers{SessionDependencies: dependencies.Session, tokenDigest: tokenDigest}
	contexts := contextHandlers{
		session:              session,
		catalog:              &catalogHandlers{CatalogDependencies: dependencies.Catalog, cursors: cursors},
		projects:             &projectHandlers{ProjectDependencies: dependencies.Projects, cursors: cursors},
		runs:                 &runHandlers{RunDependencies: dependencies.Runs, cursors: cursors},
		runRepeat:            &runRepeatHandlers{RunRepeatDependencies: dependencies.RunRepeat},
		queue:                &queueHandlers{QueueDependencies: dependencies.Queue, cursors: cursors},
		artifacts:            &artifactHandlers{ArtifactDependencies: dependencies.Artifacts, cursors: cursors},
		audits:               &auditHandlers{AuditDependencies: dependencies.Audits, cursors: cursors},
		findings:             &findingHandlers{FindingDependencies: dependencies.Findings, cursors: cursors},
		evals:                &evalHandlers{EvalDependencies: dependencies.Evals, cursors: cursors},
		operations:           &operationsHandlers{OperationsDependencies: dependencies.Operations, cursors: cursors},
		credentials:          &credentialHandlers{CredentialDependencies: dependencies.Credentials, cursors: cursors},
		runtimeConfiguration: &runtimeConfigurationHandlers{RuntimeConfigurationDependencies: dependencies.RuntimeConfiguration, cursors: cursors},
		gitKeys:              &gitKeyHandlers{GitKeyDependencies: dependencies.GitKeys},
	}

	mux := http.NewServeMux()
	contexts.registerSettingsRoutes(mux)
	contexts.registerSessionRoutes(mux)
	contexts.registerCatalogRoutes(mux)
	contexts.registerProjectRoutes(mux)
	contexts.registerAuditRoutes(mux)
	contexts.registerEvalRoutes(mux)
	contexts.registerOperationsRoutes(mux)
	contexts.registerArtifactRoutes(mux)
	contexts.registerRunRoutes(mux)
	contexts.registerQueueRoutes(mux)
	mux.HandleFunc("/", notFound)

	// Incoming requests pass API version, request ID, origin policy,
	// authentication, Operations authorization and error diagnostics in that
	// order.
	diagnostics := withErrorDiagnostics(mux)
	authenticated := session.authenticate(requireOperationsRoutes(diagnostics))
	originChecked := session.cors(authenticated, mux)
	return withAPIVersion(requestid.Middleware(originChecked, requestid.Options{
		Generator: dependencies.NewRequestID,
		Logger:    dependencies.Logger, Boundary: "public-api",
	})), nil
}

// requireServices reports the first missing service of a context. Every
// interface, pointer, function, map, slice and channel field of a context's
// dependencies is a required service; a value field is a setting whose zero
// value is valid unless NewHandler checks it.
func requireServices(context string, dependencies any) error {
	value := reflect.ValueOf(dependencies)
	for index := range value.NumField() {
		field := value.Field(index)
		switch field.Kind() {
		case reflect.Interface, reflect.Pointer, reflect.Func, reflect.Map, reflect.Slice, reflect.Chan:
			if field.IsNil() {
				return fmt.Errorf(
					"public API %s dependency %s is missing", context, value.Type().Field(index).Name,
				)
			}
		}
	}
	return nil
}
