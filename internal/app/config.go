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

	"github.com/grauwolf32/contractor/internal/auth"
	"github.com/grauwolf32/contractor/internal/contracts"
)

type repeatedStringFlag struct{ values []string }

func (f *repeatedStringFlag) String() string { return strings.Join(f.values, ",") }
func (f *repeatedStringFlag) Set(value string) error {
	if value == "" {
		return errors.New("value must not be empty")
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
	workerRequestTimeout := defaultWorkerRequestTimeout
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
	localAuthFile := getenv("CONTRACTOR_LOCAL_AUTH_FILE")
	browserOrigins := repeatedStringFlag{}
	if encoded := getenv("CONTRACTOR_BROWSER_ORIGINS"); encoded != "" {
		browserOrigins.values = strings.Split(encoded, ",")
	}
	insecureLoopbackCookie := false
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

	return Config{
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
