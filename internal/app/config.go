package app

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"net/url"
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
	databaseURL := getenv("CONTRACTOR_DATABASE_URL")
	configRoot := getenv("CONTRACTOR_CONFIG_ROOT")
	if configRoot == "" {
		configRoot = defaultConfigRoot
	}
	publicUserID := getenv("CONTRACTOR_PUBLIC_USER_ID")
	publicBearerToken := contracts.NewSecretString(getenv("CONTRACTOR_PUBLIC_BEARER_TOKEN"))
	caFile := getenv("CONTRACTOR_CA_FILE")
	certificateFile := getenv("CONTRACTOR_CONTROL_PLANE_CERT_FILE")
	privateKeyFile := getenv("CONTRACTOR_CONTROL_PLANE_KEY_FILE")
	llmGatewayURL := getenv("CONTRACTOR_LLM_GATEWAY_URL")
	llmGatewayToken := contracts.NewSecretString(getenv("CONTRACTOR_LLM_GATEWAY_TOKEN"))

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
	flags.StringVar(&configRoot, "config-root", configRoot, "configuration root")
	flags.StringVar(&publicUserID, "public-user-id", publicUserID, "single-user public API identity")
	flags.StringVar(&caFile, "ca-file", caFile, "deployment CA certificate")
	flags.StringVar(&certificateFile, "certificate-file", certificateFile, "Control Plane certificate")
	flags.StringVar(&privateKeyFile, "private-key-file", privateKeyFile, "Control Plane private key")
	flags.StringVar(&llmGatewayURL, "llm-gateway-url", llmGatewayURL, "LLM Gateway base URL")
	if err := flags.Parse(args); err != nil {
		return Config{}, fmt.Errorf("parse serve flags: %w", err)
	}
	if flags.NArg() != 0 {
		return Config{}, fmt.Errorf("unexpected positional arguments: %v", flags.Args())
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
	if strings.TrimSpace(configRoot) == "" {
		return Config{}, errors.New("configuration root must not be empty")
	}

	return Config{
		ListenAddress: listenAddress, PrivateListenAddress: privateListenAddress, PrivateURL: privateURL,
		ShutdownTimeout: shutdownTimeout, RuntimeRequestTimeout: runtimeRequestTimeout,
		DatabaseURL: databaseURL, ConfigRoot: configRoot,
		CAFile: caFile, CertificateFile: certificateFile, PrivateKeyFile: privateKeyFile,
		LLMGatewayURL: llmGatewayURL, LLMGatewayToken: llmGatewayToken,
		PublicUserID: publicUserID, PublicBearerToken: publicBearerToken,
	}, nil
}
