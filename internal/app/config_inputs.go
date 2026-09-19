package app

import (
	"errors"
	"github.com/grauwolf32/contractor/internal/contracts"
	"strconv"
	"strings"
	"time"
)

// serveConfigInputs is transient parse state. Unlike ServerConfig, it may hold
// sealed environment credentials; no serialization or logging is provided.
type serveConfigInputs struct {
	settings                    serverConfigValues
	serverConfigPath            string
	listenAddress               string
	privateListenAddress        string
	privateURL                  string
	shutdownTimeout             time.Duration
	runtimeRequestTimeout       time.Duration
	workerRequestTimeout        time.Duration
	plannerTimeout              time.Duration
	databaseURL                 string
	gitRemotes                  repeatedStringFlag
	gitKnownHosts               string
	blobBackend                 string
	blobPath                    string
	operatorConfigRoot          string
	managedConfigRoot           string
	publicUserID                string
	publicBearerToken           contracts.SecretString
	localAuthFile               string
	browserOrigins              repeatedStringFlag
	insecureLoopbackCookie      bool
	caFile                      string
	certificateFile             string
	privateKeyFile              string
	developmentWorkerToken      contracts.SecretString
	developmentPlannerToken     contracts.SecretString
	performanceMetrics          deferredBooleanFlag
	pprof                       deferredBooleanFlag
	pprofListen                 string
	credentialMasterKeyFile     string
	llmGatewayAdminBindingsFile string
	developmentLLMGateway       string
}

func loadServeConfigInputs(args []string, getenv func(string) string) (*serveConfigInputs, error) {
	serverConfigPath, err := discoverServerConfigPath(args, getenv)
	if err != nil {
		return nil, err
	}
	settings := defaultServerConfigValues()
	if serverConfigPath != "" {
		settings, err = loadServerConfig(serverConfigPath, settings)
		if err != nil {
			return nil, err
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
			return nil, errors.New("CONTRACTOR_INSECURE_LOOPBACK_COOKIE must be true or false")
		}
		parsed, err := strconv.ParseBool(encoded)
		if err != nil {
			return nil, errors.New("CONTRACTOR_INSECURE_LOOPBACK_COOKIE must be true or false")
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

	developmentLLMGateway := settings.developmentLLMGateway
	if value := getenv("CONTRACTOR_DEVELOPMENT_LLM_GATEWAY"); value != "" {
		developmentLLMGateway = value
	}
	if err := settings.operations.applyEnvironment(getenv); err != nil {
		return nil, err
	}
	return &serveConfigInputs{
		settings:                    settings,
		serverConfigPath:            serverConfigPath,
		listenAddress:               listenAddress,
		privateListenAddress:        privateListenAddress,
		privateURL:                  privateURL,
		shutdownTimeout:             shutdownTimeout,
		runtimeRequestTimeout:       runtimeRequestTimeout,
		workerRequestTimeout:        workerRequestTimeout,
		plannerTimeout:              plannerTimeout,
		databaseURL:                 databaseURL,
		gitRemotes:                  gitRemotes,
		gitKnownHosts:               gitKnownHosts,
		blobBackend:                 blobBackend,
		blobPath:                    blobPath,
		operatorConfigRoot:          operatorConfigRoot,
		managedConfigRoot:           managedConfigRoot,
		publicUserID:                publicUserID,
		publicBearerToken:           publicBearerToken,
		localAuthFile:               localAuthFile,
		browserOrigins:              browserOrigins,
		insecureLoopbackCookie:      insecureLoopbackCookie,
		caFile:                      caFile,
		certificateFile:             certificateFile,
		privateKeyFile:              privateKeyFile,
		developmentWorkerToken:      developmentWorkerToken,
		developmentPlannerToken:     developmentPlannerToken,
		performanceMetrics:          performanceMetrics,
		pprof:                       pprof,
		pprofListen:                 pprofListen,
		credentialMasterKeyFile:     credentialMasterKeyFile,
		llmGatewayAdminBindingsFile: llmGatewayAdminBindingsFile,
		developmentLLMGateway:       developmentLLMGateway,
	}, nil
}
