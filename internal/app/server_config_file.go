package app

import (
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"go.yaml.in/yaml/v4"
)

const (
	serverConfigAPIVersion = "contractor/v1alpha1"
	serverConfigKind       = "ServerConfig"
	maximumServerConfig    = 64 * 1024
)

// serverConfigValues is the non-secret process configuration after defaults
// and an optional ServerConfig document have been applied. Environment and
// command-line settings are layered over it by ParseConfig.
type serverConfigValues struct {
	operations                  OperationalSettings
	gitAllowedRemotes           []string
	gitKnownHostsFile           string
	listenAddress               string
	privateListenAddress        string
	privateURL                  string
	shutdownTimeout             time.Duration
	runtimeRequestTimeout       time.Duration
	workerRequestTimeout        time.Duration
	plannerTimeout              time.Duration
	artifactBlobBackend         string
	artifactBlobPath            string
	operatorConfigRoot          string
	managedConfigRoot           string
	credentialMasterKeyFile     string
	llmGatewayAdminBindingsFile string
	localAuthFile               string
	browserOrigins              []string
	insecureLoopbackCookie      bool
	caFile                      string
	certificateFile             string
	privateKeyFile              string
	performanceMetrics          bool
	pprof                       bool
	pprofListen                 string
}

type serverConfigDocument struct {
	APIVersion string           `yaml:"apiVersion"`
	Kind       string           `yaml:"kind"`
	Spec       serverConfigSpec `yaml:"spec"`
}

// Pointer leaves distinguish an omitted setting from an explicit false or
// empty value. Secret bytes and connection URLs intentionally have no fields.
type serverConfigSpec struct {
	operationalSpec             `yaml:",inline"`
	GitAllowedRemotes           *[]string `yaml:"gitAllowedRemotes"`
	GitKnownHostsFile           *string   `yaml:"gitKnownHostsFile"`
	Listen                      *string   `yaml:"listen"`
	PrivateListen               *string   `yaml:"privateListen"`
	PrivateURL                  *string   `yaml:"privateUrl"`
	ShutdownTimeout             *string   `yaml:"shutdownTimeout"`
	RuntimeRequestTimeout       *string   `yaml:"runtimeRequestTimeout"`
	WorkerRequestTimeout        *string   `yaml:"workerRequestTimeout"`
	PlannerTimeout              *string   `yaml:"plannerTimeout"`
	ArtifactBlobBackend         *string   `yaml:"artifactBlobBackend"`
	ArtifactBlobPath            *string   `yaml:"artifactBlobPath"`
	OperatorConfigRoot          *string   `yaml:"operatorConfigRoot"`
	ManagedConfigRoot           *string   `yaml:"managedConfigRoot"`
	CredentialMasterKeyFile     *string   `yaml:"credentialMasterKeyFile"`
	LLMGatewayAdminBindingsFile *string   `yaml:"llmGatewayAdminBindingsFile"`
	LocalAuthFile               *string   `yaml:"localAuthFile"`
	BrowserOrigins              *[]string `yaml:"browserOrigins"`
	InsecureLoopbackCookie      *bool     `yaml:"insecureLoopbackCookie"`
	CAFile                      *string   `yaml:"caFile"`
	CertificateFile             *string   `yaml:"certificateFile"`
	PrivateKeyFile              *string   `yaml:"privateKeyFile"`
	PerformanceMetrics          *bool     `yaml:"performanceMetrics"`
	Pprof                       *bool     `yaml:"pprof"`
	PprofListen                 *string   `yaml:"pprofListen"`
}

func defaultServerConfigValues() serverConfigValues {
	return serverConfigValues{
		operations:            defaultOperationalSettings(),
		listenAddress:         defaultListenAddress,
		privateListenAddress:  defaultPrivateListenAddress,
		privateURL:            defaultPrivateURL,
		shutdownTimeout:       defaultShutdownTimeout,
		runtimeRequestTimeout: defaultRuntimeRequestTimeout,
		workerRequestTimeout:  defaultWorkerRequestTimeout,
		plannerTimeout:        defaultPlannerTimeout,
		operatorConfigRoot:    defaultConfigRoot,
		performanceMetrics:    true,
		pprofListen:           "127.0.0.1:6060",
	}
}

// discoverServerConfigPath performs only the bootstrap parsing needed before
// the complete FlagSet exists. The same aliases are registered on that FlagSet
// later, so ordinary flag validation remains authoritative.
func discoverServerConfigPath(args []string, getenv func(string) string) (string, error) {
	path := getenv("CONTRACTOR_SERVER_CONFIG")
	explicit := path != ""
	for index := 0; index < len(args); index++ {
		argument := args[index]
		for _, name := range []string{"config", "server-config"} {
			for _, prefix := range []string{"--" + name + "=", "-" + name + "="} {
				if strings.HasPrefix(argument, prefix) {
					path = strings.TrimPrefix(argument, prefix)
					explicit = true
				}
			}
			if argument == "--"+name || argument == "-"+name {
				if index+1 >= len(args) {
					return "", fmt.Errorf("%s requires a file path", argument)
				}
				index++
				path = args[index]
				explicit = true
			}
		}
	}
	if explicit && strings.TrimSpace(path) == "" {
		return "", errors.New("server config path must not be empty")
	}
	return path, nil
}

func loadServerConfig(path string, values serverConfigValues) (serverConfigValues, error) {
	absolute, err := filepath.Abs(path)
	if err != nil {
		return serverConfigValues{}, fmt.Errorf("resolve server config path: %w", err)
	}
	file, err := os.Open(absolute)
	if err != nil {
		return serverConfigValues{}, fmt.Errorf("open server config: %w", err)
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil || !info.Mode().IsRegular() || info.Size() < 1 || info.Size() > maximumServerConfig {
		return serverConfigValues{}, errors.New("server config must be a non-empty regular file no larger than 64 KiB")
	}
	data, err := io.ReadAll(io.LimitReader(file, maximumServerConfig+1))
	if err != nil || len(data) == 0 || len(data) > maximumServerConfig {
		return serverConfigValues{}, errors.New("read bounded server config")
	}
	var documents []serverConfigDocument
	if err := yaml.Load(
		data,
		&documents,
		yaml.WithAllDocuments(),
		yaml.WithKnownFields(),
		yaml.WithUniqueKeys(),
	); err != nil {
		return serverConfigValues{}, fmt.Errorf("decode strict server config YAML: %w", err)
	}
	if len(documents) != 1 {
		return serverConfigValues{}, fmt.Errorf("server config must contain exactly one YAML document, got %d", len(documents))
	}
	document := documents[0]
	if document.APIVersion != serverConfigAPIVersion {
		return serverConfigValues{}, fmt.Errorf("server config apiVersion must be %q", serverConfigAPIVersion)
	}
	if document.Kind != serverConfigKind {
		return serverConfigValues{}, fmt.Errorf("server config kind must be %q", serverConfigKind)
	}
	if err := applyServerConfigSpec(&values, document.Spec, filepath.Dir(absolute)); err != nil {
		return serverConfigValues{}, err
	}
	return values, nil
}

func applyServerConfigSpec(values *serverConfigValues, spec serverConfigSpec, base string) error {
	if err := values.operations.applySpec(spec.operationalSpec); err != nil {
		return err
	}
	setString(&values.listenAddress, spec.Listen)
	setString(&values.privateListenAddress, spec.PrivateListen)
	setString(&values.privateURL, spec.PrivateURL)
	if err := setDuration(&values.shutdownTimeout, spec.ShutdownTimeout, "shutdownTimeout"); err != nil {
		return err
	}
	if err := setDuration(&values.runtimeRequestTimeout, spec.RuntimeRequestTimeout, "runtimeRequestTimeout"); err != nil {
		return err
	}
	if err := setDuration(&values.workerRequestTimeout, spec.WorkerRequestTimeout, "workerRequestTimeout"); err != nil {
		return err
	}
	if err := setDuration(&values.plannerTimeout, spec.PlannerTimeout, "plannerTimeout"); err != nil {
		return err
	}
	if spec.GitAllowedRemotes != nil {
		values.gitAllowedRemotes = append([]string(nil), (*spec.GitAllowedRemotes)...)
	}
	setConfigPath(&values.gitKnownHostsFile, spec.GitKnownHostsFile, base)
	setString(&values.artifactBlobBackend, spec.ArtifactBlobBackend)
	setConfigPath(&values.artifactBlobPath, spec.ArtifactBlobPath, base)
	setConfigPath(&values.operatorConfigRoot, spec.OperatorConfigRoot, base)
	setConfigPath(&values.managedConfigRoot, spec.ManagedConfigRoot, base)
	setConfigPath(&values.credentialMasterKeyFile, spec.CredentialMasterKeyFile, base)
	setConfigPath(&values.llmGatewayAdminBindingsFile, spec.LLMGatewayAdminBindingsFile, base)
	setConfigPath(&values.localAuthFile, spec.LocalAuthFile, base)
	if spec.BrowserOrigins != nil {
		values.browserOrigins = append([]string(nil), (*spec.BrowserOrigins)...)
	}
	if spec.InsecureLoopbackCookie != nil {
		values.insecureLoopbackCookie = *spec.InsecureLoopbackCookie
	}
	setConfigPath(&values.caFile, spec.CAFile, base)
	setConfigPath(&values.certificateFile, spec.CertificateFile, base)
	setConfigPath(&values.privateKeyFile, spec.PrivateKeyFile, base)
	if spec.PerformanceMetrics != nil {
		values.performanceMetrics = *spec.PerformanceMetrics
	}
	if spec.Pprof != nil {
		values.pprof = *spec.Pprof
	}
	setString(&values.pprofListen, spec.PprofListen)
	return nil
}

func setString(target *string, source *string) {
	if source != nil {
		*target = *source
	}
}

func setConfigPath(target *string, source *string, base string) {
	if source == nil {
		return
	}
	if *source == "" {
		*target = ""
		return
	}
	if filepath.IsAbs(*source) {
		*target = filepath.Clean(*source)
		return
	}
	*target = filepath.Clean(filepath.Join(base, *source))
}

func setDuration(target *time.Duration, source *string, name string) error {
	if source == nil {
		return nil
	}
	parsed, err := time.ParseDuration(*source)
	if err != nil {
		return fmt.Errorf("server config spec.%s must be a duration", name)
	}
	*target = parsed
	return nil
}
