package app

import (
	"errors"
	"fmt"
	"net"
	"strings"
	"time"
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
	defaultDevelopmentLLMGateway = "local-litellm@1"
)

// ParseConfig parses the serve command without reading global process state,
// which keeps tests isolated and prevents accidental environment logging.
func ParseConfig(args []string, getenv func(string) string) (Config, error) {
	if len(args) > 0 && args[0] == "serve" {
		args = args[1:]
	} else if len(args) > 0 && args[0] != "serve" && !strings.HasPrefix(args[0], "-") {
		return Config{}, fmt.Errorf("unknown command %q", args[0])
	}

	inputs, err := loadServeConfigInputs(args, getenv)
	if err != nil {
		return Config{}, err
	}
	if err := inputs.parseFlags(args); err != nil {
		return Config{}, err
	}
	return inputs.effectiveConfig()
}

func isLoopbackListenAddress(address string) bool {
	host, _, err := net.SplitHostPort(address)
	if err != nil {
		return false
	}
	parsed := net.ParseIP(host)
	return parsed != nil && parsed.IsLoopback()
}
