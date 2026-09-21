package app

import (
	"errors"
	"fmt"
	"net/url"
	"path/filepath"
	"strings"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auth"
	workflowconfig "github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/gitimport"
)

// effectiveConfig applies derived defaults and validates the final flag-overridden
// values. Deferred booleans are intentionally parsed here, after CLI overrides.
func (c *serveConfigInputs) effectiveConfig() (Config, error) {
	if err := c.settings.operations.validate(); err != nil {
		return Config{}, err
	}
	metricsEnabled, err := c.performanceMetrics.parse("performance-metrics")
	if err != nil {
		return Config{}, err
	}
	pprofEnabled, err := c.pprof.parse("pprof")
	if err != nil {
		return Config{}, err
	}
	if !validPprofListen(c.pprofListen) {
		return Config{}, errors.New("pprof-listen must be a numeric loopback IP and TCP port between 1 and 65535")
	}
	if c.managedConfigRoot == "" {
		c.managedConfigRoot = filepath.Join(filepath.Dir(c.operatorConfigRoot), "managed-configs")
	}
	if c.listenAddress == "" {
		return Config{}, errors.New("listen address must not be empty")
	}
	if c.privateListenAddress == "" || strings.TrimSpace(c.privateURL) == "" {
		return Config{}, errors.New("private listen address and URL must not be empty")
	}
	parsedPrivateURL, err := url.Parse(c.privateURL)
	if err != nil || parsedPrivateURL.Scheme != "https" || parsedPrivateURL.Host == "" ||
		parsedPrivateURL.User != nil || parsedPrivateURL.RawQuery != "" || parsedPrivateURL.Fragment != "" ||
		(parsedPrivateURL.Path != "" && parsedPrivateURL.Path != "/") {
		return Config{}, errors.New("private URL must be an HTTPS origin without credentials, query, fragment, or path")
	}
	if c.shutdownTimeout <= 0 {
		return Config{}, errors.New("shutdown timeout must be positive")
	}
	if c.runtimeRequestTimeout < time.Second || c.runtimeRequestTimeout%time.Second != 0 {
		return Config{}, errors.New("runtime request timeout must be positive whole seconds")
	}
	if c.workerRequestTimeout < minimumWorkerRequestTimeout || c.workerRequestTimeout%time.Second != 0 {
		return Config{}, errors.New("worker request timeout must be at least 120 whole seconds")
	}
	if c.plannerTimeout <= 0 || c.plannerTimeout > 30*time.Minute {
		return Config{}, errors.New("planner timeout must be positive and at most 30 minutes")
	}
	if strings.TrimSpace(c.operatorConfigRoot) == "" || strings.TrimSpace(c.managedConfigRoot) == "" {
		return Config{}, errors.New("operator and managed configuration roots must not be empty")
	}
	if c.insecureLoopbackCookie && !isLoopbackListenAddress(c.listenAddress) {
		return Config{}, errors.New("insecure loopback cookie requires an IP-literal loopback public listener")
	}
	if len(c.browserOrigins.values) != 0 {
		if _, err := auth.NewOriginPolicy(c.browserOrigins.values, c.insecureLoopbackCookie); err != nil {
			return Config{}, fmt.Errorf("invalid browser origins: %w", err)
		}
	}
	if _, err := auth.NewPeerPolicy(c.trustedProxies.values); err != nil {
		return Config{}, fmt.Errorf("invalid trusted proxies: %w", err)
	}
	if _, err := workflowconfig.ParseSelector(c.developmentLLMGateway); err != nil {
		return Config{}, errors.New("development-llm-gateway must be an exact name@version selector")
	}
	if c.developmentPlannerToken.Reveal() == "" {
		c.developmentPlannerToken = c.developmentWorkerToken
	}

	selectedBlobBackend, err := artifacts.ValidateBlobConfig(c.blobBackend, c.blobPath)
	if err != nil {
		return Config{}, err
	}
	gitConfig := gitimport.Config{AllowedRemotes: c.gitRemotes.values, KnownHostsFile: c.gitKnownHosts}
	if err := gitimport.ValidateConfig(gitConfig); err != nil {
		return Config{}, err
	}
	return Config{
		Operations:          c.settings.operations,
		GitImport:           gitConfig,
		ArtifactBlobBackend: selectedBlobBackend, ArtifactBlobPath: c.blobPath,
		ListenAddress: c.listenAddress, PrivateListenAddress: c.privateListenAddress, PrivateURL: c.privateURL,
		ShutdownTimeout: c.shutdownTimeout, RuntimeRequestTimeout: c.runtimeRequestTimeout,
		WorkerRequestTimeout:        c.workerRequestTimeout,
		DatabaseURL:                 c.databaseURL,
		OperatorConfigRoot:          c.operatorConfigRoot,
		ManagedConfigRoot:           c.managedConfigRoot,
		CredentialMasterKeyFile:     c.credentialMasterKeyFile,
		LLMGatewayAdminBindingsFile: c.llmGatewayAdminBindingsFile,
		CAFile:                      c.caFile, CertificateFile: c.certificateFile, PrivateKeyFile: c.privateKeyFile,
		DevelopmentLLMGateway:  c.developmentLLMGateway,
		DevelopmentWorkerToken: c.developmentWorkerToken, DevelopmentPlannerToken: c.developmentPlannerToken,
		PlannerTimeout:    c.plannerTimeout,
		PublicBearerToken: c.publicBearerToken,
		LocalAuthFile:     c.localAuthFile, BrowserOrigins: append([]string(nil), c.browserOrigins.values...),
		InsecureLoopbackCookie: c.insecureLoopbackCookie,
		TrustedProxies:         append([]string(nil), c.trustedProxies.values...),
		PerformanceMetrics:     metricsEnabled, Pprof: pprofEnabled, PprofListen: c.pprofListen,
	}, nil
}
