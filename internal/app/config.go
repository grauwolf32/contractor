package app

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"net/url"
	"path/filepath"
	"strings"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
)

const (
	defaultListenAddress         = "127.0.0.1:8080"
	defaultPrivateListenAddress  = "127.0.0.1:8443"
	defaultPrivateURL            = "https://127.0.0.1:8443"
	defaultShutdownTimeout       = 5 * time.Second
	defaultRuntimeRequestTimeout = 30 * time.Second
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

	listenAddress := getenv("CONTRACTOR_PUBLIC_LISTEN")
	if listenAddress == "" {
		listenAddress = defaultListenAddress
	}
	privateListenAddress := getenv("CONTRACTOR_PRIVATE_LISTEN")
	if privateListenAddress == "" {
		privateListenAddress = defaultPrivateListenAddress
	}
	privateURL := getenv("CONTRACTOR_PRIVATE_URL")
	if privateURL == "" {
		privateURL = defaultPrivateURL
	}
	shutdownTimeout := defaultShutdownTimeout
	runtimeRequestTimeout := defaultRuntimeRequestTimeout
	plannerTimeout := defaultPlannerTimeout
	databaseURL := getenv("CONTRACTOR_DATABASE_URL")
	operatorConfigRoot := getenv("CONTRACTOR_OPERATOR_CONFIG_ROOT")
	if operatorConfigRoot == "" {
		operatorConfigRoot = getenv("CONTRACTOR_CONFIG_ROOT")
	}
	if operatorConfigRoot == "" {
		operatorConfigRoot = defaultConfigRoot
	}
	managedConfigRoot := getenv("CONTRACTOR_MANAGED_CONFIG_ROOT")
	publicUserID := getenv("CONTRACTOR_PUBLIC_USER_ID")
	publicBearerToken := contracts.NewSecretString(getenv("CONTRACTOR_PUBLIC_BEARER_TOKEN"))
	caFile := getenv("CONTRACTOR_CA_FILE")
	certificateFile := getenv("CONTRACTOR_CONTROL_PLANE_CERT_FILE")
	privateKeyFile := getenv("CONTRACTOR_CONTROL_PLANE_KEY_FILE")
	developmentWorkerToken := contracts.NewSecretString(getenv("CONTRACTOR_LLM_GATEWAY_TOKEN"))
	developmentPlannerToken := contracts.NewSecretString(getenv("CONTRACTOR_PLANNER_LLM_GATEWAY_TOKEN"))

	flags := flag.NewFlagSet("contractor-server serve", flag.ContinueOnError)
	flags.SetOutput(io.Discard)
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
	flags.StringVar(&databaseURL, "database-url", databaseURL, "PostgreSQL connection URL")
	flags.StringVar(&operatorConfigRoot, "operator-config-root", operatorConfigRoot, "operator/bootstrap configuration root")
	flags.StringVar(&operatorConfigRoot, "config-root", operatorConfigRoot, "deprecated alias for --operator-config-root")
	flags.StringVar(&managedConfigRoot, "managed-config-root", managedConfigRoot, "Server-managed configuration publication root")
	credentialMasterKeyFile := ""
	flags.StringVar(
		&credentialMasterKeyFile,
		"credential-master-key-file",
		credentialMasterKeyFile,
		"absolute owner-only file containing the credential encryption key",
	)
	llmGatewayAdminBindingsFile := ""
	flags.StringVar(
		&llmGatewayAdminBindingsFile,
		"llm-gateway-admin-bindings-file",
		llmGatewayAdminBindingsFile,
		"absolute strict YAML file binding exact LLM Gateways to owner-only admin-key files",
	)
	flags.StringVar(&publicUserID, "public-user-id", publicUserID, "single-user public API identity")
	flags.StringVar(&caFile, "ca-file", caFile, "deployment CA certificate")
	flags.StringVar(&certificateFile, "certificate-file", certificateFile, "Control Plane certificate")
	flags.StringVar(&privateKeyFile, "private-key-file", privateKeyFile, "Control Plane private key")
	flags.DurationVar(&plannerTimeout, "planner-timeout", plannerTimeout, "maximum Planner wall time")
	if err := flags.Parse(args); err != nil {
		return Config{}, fmt.Errorf("parse serve flags: %w", err)
	}
	if flags.NArg() != 0 {
		return Config{}, fmt.Errorf("unexpected positional arguments: %v", flags.Args())
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
	if plannerTimeout <= 0 || plannerTimeout > 30*time.Minute {
		return Config{}, errors.New("planner timeout must be positive and at most 30 minutes")
	}
	if strings.TrimSpace(operatorConfigRoot) == "" || strings.TrimSpace(managedConfigRoot) == "" {
		return Config{}, errors.New("operator and managed configuration roots must not be empty")
	}
	if developmentPlannerToken.Reveal() == "" {
		developmentPlannerToken = developmentWorkerToken
	}

	return Config{
		ListenAddress: listenAddress, PrivateListenAddress: privateListenAddress, PrivateURL: privateURL,
		ShutdownTimeout: shutdownTimeout, RuntimeRequestTimeout: runtimeRequestTimeout,
		DatabaseURL: databaseURL,
		ConfigRoot:  operatorConfigRoot, OperatorConfigRoot: operatorConfigRoot,
		ManagedConfigRoot:           managedConfigRoot,
		CredentialMasterKeyFile:     credentialMasterKeyFile,
		LLMGatewayAdminBindingsFile: llmGatewayAdminBindingsFile,
		CAFile:                      caFile, CertificateFile: certificateFile, PrivateKeyFile: privateKeyFile,
		DevelopmentWorkerToken: developmentWorkerToken, DevelopmentPlannerToken: developmentPlannerToken,
		PlannerTimeout: plannerTimeout,
		PublicUserID:   publicUserID, PublicBearerToken: publicBearerToken,
	}, nil
}
