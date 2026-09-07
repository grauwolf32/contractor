package app

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"net"
	"net/url"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auth"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/gitimport"
)

type repeatedStringFlag struct {
	values          []string
	clearOnFirstSet bool
}

func (f *repeatedStringFlag) String() string { return strings.Join(f.values, ",") }
func (f *repeatedStringFlag) Set(value string) error {
	if value == "" {
		return errors.New("value must not be empty")
	}
	if f.clearOnFirstSet {
		f.values = nil
		f.clearOnFirstSet = false
	}
	f.values = append(f.values, value)
	return nil
}

const (
	defaultListenAddress         = "127.0.0.1:8080"
	defaultPrivateListenAddress  = "127.0.0.1:8443"
	defaultPrivateURL            = "https://127.0.0.1:8443"
	defaultShutdownTimeout       = 5 * time.Second
	defaultRuntimeRequestTimeout = 30 * time.Second
	minimumWorkerRequestTimeout  = 120 * time.Second
	defaultWorkerRequestTimeout  = 180 * time.Second
	defaultPlannerTimeout        = 30 * time.Minute
	defaultConfigRoot            = "./configs"
)

// ParseConfig parses the serve command without reading global process state,
// which keeps tests isolated and prevents accidental environment logging.
func ParseConfig(args []string, getenv func(string) string) (Config, error) {
	if len(args) > 0 && args[0] == "serve" {
		args = args[1:]
	} else if len(args) > 0 && args[0] != "serve" && !strings.HasPrefix(args[0], "-") {
		return Config{}, fmt.Errorf("unknown command %q", args[0])
	}

	serverConfigPath, err := discoverServerConfigPath(args, getenv)
	if err != nil {
		return Config{}, err
	}
	settings := defaultServerConfigValues()
	if serverConfigPath != "" {
		settings, err = loadServerConfig(serverConfigPath, settings)
		if err != nil {
			return Config{}, err
		}
	}

	listenAddress := settings.listenAddress
	if value := getenv("CONTRACTOR_PUBLIC_LISTEN"); value != "" {
		listenAddress = value
	}
	privateListenAddress := settings.privateListenAddress
	if value := getenv("CONTRACTOR_PRIVATE_LISTEN"); value != "" {
		privateListenAddress = value
	}
	privateURL := settings.privateURL
	if value := getenv("CONTRACTOR_PRIVATE_URL"); value != "" {
		privateURL = value
	}
	shutdownTimeout := settings.shutdownTimeout
	runtimeRequestTimeout := settings.runtimeRequestTimeout
	workerRequestTimeout := settings.workerRequestTimeout
	plannerTimeout := settings.plannerTimeout
	databaseURL := getenv("CONTRACTOR_DATABASE_URL")
	gitRemotes := repeatedStringFlag{values: settings.gitAllowedRemotes, clearOnFirstSet: true}
	if value := getenv("CONTRACTOR_GIT_ALLOWED_REMOTES"); value != "" {
		gitRemotes.values = strings.Split(value, ",")
		gitRemotes.clearOnFirstSet = true
	}
	gitKnownHosts := settings.gitKnownHostsFile
	if value := getenv("CONTRACTOR_GIT_KNOWN_HOSTS_FILE"); value != "" {
		gitKnownHosts = value
	}
	blobBackend := settings.artifactBlobBackend
	if value := getenv("CONTRACTOR_ARTIFACT_BLOB_BACKEND"); value != "" {
		blobBackend = value
	}
	blobPath := settings.artifactBlobPath
	if value := getenv("CONTRACTOR_ARTIFACT_BLOB_PATH"); value != "" {
		blobPath = value
	}
	operatorConfigRoot := settings.operatorConfigRoot
	if value := getenv("CONTRACTOR_OPERATOR_CONFIG_ROOT"); value != "" {
		operatorConfigRoot = value
	} else if value := getenv("CONTRACTOR_CONFIG_ROOT"); value != "" {
		operatorConfigRoot = value
	}
	managedConfigRoot := settings.managedConfigRoot
	if value := getenv("CONTRACTOR_MANAGED_CONFIG_ROOT"); value != "" {
		managedConfigRoot = value
	}
	publicUserID := getenv("CONTRACTOR_PUBLIC_USER_ID")
	publicBearerToken := contracts.NewSecretString(getenv("CONTRACTOR_PUBLIC_BEARER_TOKEN"))
	localAuthFile := settings.localAuthFile
	if value := getenv("CONTRACTOR_LOCAL_AUTH_FILE"); value != "" {
		localAuthFile = value
	}
	browserOrigins := repeatedStringFlag{
		values:          append([]string(nil), settings.browserOrigins...),
		clearOnFirstSet: true,
	}
	if encoded := getenv("CONTRACTOR_BROWSER_ORIGINS"); encoded != "" {
		browserOrigins.values = strings.Split(encoded, ",")
	}
	insecureLoopbackCookie := settings.insecureLoopbackCookie
	if encoded := getenv("CONTRACTOR_INSECURE_LOOPBACK_COOKIE"); encoded != "" {
		if encoded != "true" && encoded != "false" {
			return Config{}, errors.New("CONTRACTOR_INSECURE_LOOPBACK_COOKIE must be true or false")
		}
		parsed, err := strconv.ParseBool(encoded)
		if err != nil {
			return Config{}, errors.New("CONTRACTOR_INSECURE_LOOPBACK_COOKIE must be true or false")
		}
		insecureLoopbackCookie = parsed
	}
	caFile := settings.caFile
	if value := getenv("CONTRACTOR_CA_FILE"); value != "" {
		caFile = value
	}
	certificateFile := settings.certificateFile
	if value := getenv("CONTRACTOR_CONTROL_PLANE_CERT_FILE"); value != "" {
		certificateFile = value
	}
	privateKeyFile := settings.privateKeyFile
	if value := getenv("CONTRACTOR_CONTROL_PLANE_KEY_FILE"); value != "" {
		privateKeyFile = value
	}
	developmentWorkerToken := contracts.NewSecretString(getenv("CONTRACTOR_LLM_GATEWAY_TOKEN"))
	developmentPlannerToken := contracts.NewSecretString(getenv("CONTRACTOR_PLANNER_LLM_GATEWAY_TOKEN"))
	performanceMetrics := deferredBooleanFlag{
		value: getenv("CONTRACTOR_PERFORMANCE_METRICS"), fallback: settings.performanceMetrics,
	}
	pprof := deferredBooleanFlag{value: getenv("CONTRACTOR_PPROF"), fallback: settings.pprof}
	pprofListen := settings.pprofListen
	if value := getenv("CONTRACTOR_PPROF_LISTEN"); value != "" {
		pprofListen = value
	}
	credentialMasterKeyFile := settings.credentialMasterKeyFile
	if value := getenv("CONTRACTOR_CREDENTIAL_MASTER_KEY_FILE"); value != "" {
		credentialMasterKeyFile = value
	}
	llmGatewayAdminBindingsFile := settings.llmGatewayAdminBindingsFile
	if value := getenv("CONTRACTOR_LLM_GATEWAY_ADMIN_BINDINGS_FILE"); value != "" {
		llmGatewayAdminBindingsFile = value
	}

	flags := flag.NewFlagSet("contractor-server serve", flag.ContinueOnError)
	flags.SetOutput(io.Discard)
	if err := settings.operations.registerFlags(flags, getenv); err != nil {
		return Config{}, err
	}
	flags.Var(&gitRemotes, "git-allowed-remote", "allowed Git remote host:port; repeatable")
	flags.StringVar(&gitKnownHosts, "git-known-hosts-file", gitKnownHosts, "absolute read-only SSH known_hosts file")
	flags.StringVar(&serverConfigPath, "config", serverConfigPath, "strict YAML ServerConfig file")
	flags.StringVar(&serverConfigPath, "server-config", serverConfigPath, "alias for --config")
	flags.StringVar(&blobBackend, "artifact-blob-backend", blobBackend, "Artifact payload backend: postgresql or filesystem")
	flags.StringVar(&blobPath, "artifact-blob-path", blobPath, "absolute filesystem blob directory")
	flags.Var(&performanceMetrics, "performance-metrics", "enable Operations performance collection (requires restart)")
	flags.Var(&pprof, "pprof", "enable independent loopback Go profiling (requires restart)")
	flags.StringVar(&pprofListen, "pprof-listen", pprofListen, "numeric loopback Go profiling listen address")
	flags.StringVar(&listenAddress, "listen", listenAddress, "public HTTP listen address")
	flags.StringVar(&privateListenAddress, "private-listen", privateListenAddress, "private mTLS listen address")
	flags.StringVar(&privateURL, "private-url", privateURL, "advertised private mTLS base URL")
	flags.DurationVar(
		&shutdownTimeout,
		"shutdown-timeout",
		shutdownTimeout,
		"graceful shutdown timeout",
	)
	flags.DurationVar(
		&runtimeRequestTimeout,
		"runtime-request-timeout",
		runtimeRequestTimeout,
		"private Runtime Agent request timeout",
	)
	flags.DurationVar(
		&workerRequestTimeout,
		"worker-request-timeout",
		workerRequestTimeout,
		"allocation-local Worker outbound request timeout (minimum 120s)",
	)
	flags.StringVar(&databaseURL, "database-url", databaseURL, "PostgreSQL connection URL")
	flags.StringVar(&operatorConfigRoot, "operator-config-root", operatorConfigRoot, "operator/bootstrap configuration root")
	flags.StringVar(&operatorConfigRoot, "config-root", operatorConfigRoot, "deprecated alias for --operator-config-root")
	flags.StringVar(&managedConfigRoot, "managed-config-root", managedConfigRoot, "Server-managed configuration publication root")
	flags.StringVar(
		&credentialMasterKeyFile,
		"credential-master-key-file",
		credentialMasterKeyFile,
		"absolute owner-only file containing the credential encryption key",
	)
	flags.StringVar(
		&llmGatewayAdminBindingsFile,
		"llm-gateway-admin-bindings-file",
		llmGatewayAdminBindingsFile,
		"absolute strict YAML file binding exact LLM Gateways to owner-only admin-key files",
	)
	flags.StringVar(&publicUserID, "public-user-id", publicUserID, "deprecated assertion matching local-auth userId")
	flags.StringVar(&localAuthFile, "local-auth-file", localAuthFile, "absolute owner-only local authentication YAML")
	flags.Var(&browserOrigins, "browser-origin", "exact allowed browser UI origin; repeat for multiple origins")
	flags.BoolVar(
		&insecureLoopbackCookie,
		"insecure-loopback-cookie",
		insecureLoopbackCookie,
		"use the separately named insecure cookie on an IP-literal loopback listener",
	)
	flags.StringVar(&caFile, "ca-file", caFile, "deployment CA certificate")
	flags.StringVar(&certificateFile, "certificate-file", certificateFile, "Control Plane certificate")
	flags.StringVar(&privateKeyFile, "private-key-file", privateKeyFile, "Control Plane private key")
	flags.DurationVar(&plannerTimeout, "planner-timeout", plannerTimeout, "maximum Stage preparation and Planner wall time")
	if err := flags.Parse(args); err != nil {
		return Config{}, fmt.Errorf("parse serve flags: %w", err)
	}
	if flags.NArg() != 0 {
		return Config{}, fmt.Errorf("unexpected positional arguments: %v", flags.Args())
	}
	if err := settings.operations.validate(); err != nil {
		return Config{}, err
	}
	metricsEnabled, err := performanceMetrics.parse("performance-metrics")
	if err != nil {
		return Config{}, err
	}
	pprofEnabled, err := pprof.parse("pprof")
	if err != nil {
		return Config{}, err
	}
	if !validPprofListen(pprofListen) {
		return Config{}, errors.New("pprof-listen must be a numeric loopback IP and TCP port between 1 and 65535")
	}
	if managedConfigRoot == "" {
		managedConfigRoot = filepath.Join(filepath.Dir(operatorConfigRoot), "managed-configs")
	}
	if listenAddress == "" {
		return Config{}, errors.New("listen address must not be empty")
	}
	if privateListenAddress == "" || strings.TrimSpace(privateURL) == "" {
		return Config{}, errors.New("private listen address and URL must not be empty")
	}
	parsedPrivateURL, err := url.Parse(privateURL)
	if err != nil || parsedPrivateURL.Scheme != "https" || parsedPrivateURL.Host == "" ||
		parsedPrivateURL.User != nil || parsedPrivateURL.RawQuery != "" || parsedPrivateURL.Fragment != "" ||
		(parsedPrivateURL.Path != "" && parsedPrivateURL.Path != "/") {
		return Config{}, errors.New("private URL must be an HTTPS origin without credentials, query, fragment, or path")
	}
	if shutdownTimeout <= 0 {
		return Config{}, errors.New("shutdown timeout must be positive")
	}
	if runtimeRequestTimeout < time.Second || runtimeRequestTimeout%time.Second != 0 {
		return Config{}, errors.New("runtime request timeout must be positive whole seconds")
	}
	if workerRequestTimeout < minimumWorkerRequestTimeout || workerRequestTimeout%time.Second != 0 {
		return Config{}, errors.New("worker request timeout must be at least 120 whole seconds")
	}
	if plannerTimeout <= 0 || plannerTimeout > 30*time.Minute {
		return Config{}, errors.New("planner timeout must be positive and at most 30 minutes")
	}
	if strings.TrimSpace(operatorConfigRoot) == "" || strings.TrimSpace(managedConfigRoot) == "" {
		return Config{}, errors.New("operator and managed configuration roots must not be empty")
	}
	if insecureLoopbackCookie && !isLoopbackListenAddress(listenAddress) {
		return Config{}, errors.New("insecure loopback cookie requires an IP-literal loopback public listener")
	}
	if len(browserOrigins.values) != 0 {
		if _, err := auth.NewOriginPolicy(browserOrigins.values, insecureLoopbackCookie); err != nil {
			return Config{}, fmt.Errorf("invalid browser origins: %w", err)
		}
	}
	if developmentPlannerToken.Reveal() == "" {
		developmentPlannerToken = developmentWorkerToken
	}

	selectedBlobBackend, err := artifacts.ValidateBlobConfig(blobBackend, blobPath)
	if err != nil {
		return Config{}, err
	}
	gitConfig := gitimport.Config{AllowedRemotes: gitRemotes.values, KnownHostsFile: gitKnownHosts}
	if err := gitimport.ValidateConfig(gitConfig); err != nil {
		return Config{}, err
	}
	return Config{
		Operations:          settings.operations,
		GitImport:           gitConfig,
		ArtifactBlobBackend: selectedBlobBackend, ArtifactBlobPath: blobPath,
		ListenAddress: listenAddress, PrivateListenAddress: privateListenAddress, PrivateURL: privateURL,
		ShutdownTimeout: shutdownTimeout, RuntimeRequestTimeout: runtimeRequestTimeout,
		WorkerRequestTimeout: workerRequestTimeout,
		DatabaseURL:          databaseURL,
		ConfigRoot:           operatorConfigRoot, OperatorConfigRoot: operatorConfigRoot,
		ManagedConfigRoot:           managedConfigRoot,
		CredentialMasterKeyFile:     credentialMasterKeyFile,
		LLMGatewayAdminBindingsFile: llmGatewayAdminBindingsFile,
		CAFile:                      caFile, CertificateFile: certificateFile, PrivateKeyFile: privateKeyFile,
		DevelopmentWorkerToken: developmentWorkerToken, DevelopmentPlannerToken: developmentPlannerToken,
		PlannerTimeout: plannerTimeout,
		PublicUserID:   publicUserID, PublicBearerToken: publicBearerToken,
		LocalAuthFile: localAuthFile, BrowserOrigins: append([]string(nil), browserOrigins.values...),
		InsecureLoopbackCookie: insecureLoopbackCookie,
		PerformanceMetrics:     metricsEnabled, Pprof: pprofEnabled, PprofListen: pprofListen,
	}, nil
}

func isLoopbackListenAddress(address string) bool {
	host, _, err := net.SplitHostPort(address)
	if err != nil {
		return false
	}
	parsed := net.ParseIP(host)
	return parsed != nil && parsed.IsLoopback()
}
