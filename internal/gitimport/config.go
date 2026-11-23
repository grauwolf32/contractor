package gitimport

import (
	"net"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/grauwolf32/contractor/internal/apierror"
)

var ErrConfiguration = apierror.New(apierror.GitImportUnavailable, "invalid Git remote allowlist or known_hosts path")

type Config struct {
	Limits         *Limits
	AllowedRemotes []string
	KnownHostsFile string
}

func ValidateConfig(cfg Config) error {
	if err := cfg.EffectiveLimits().Validate(); err != nil {
		return err
	}
	if cfg.KnownHostsFile != "" && (!filepath.IsAbs(cfg.KnownHostsFile) || filepath.Clean(cfg.KnownHostsFile) != cfg.KnownHostsFile) {
		return ErrConfiguration
	}
	if len(cfg.AllowedRemotes) > 100 {
		return ErrConfiguration
	}
	seen := map[string]bool{}
	for _, remote := range cfg.AllowedRemotes {
		host, port, err := net.SplitHostPort(remote)
		n, pErr := strconv.Atoi(port)
		if err != nil || pErr != nil || n < 1 || n > 65535 || port != strconv.Itoa(n) || host == "" || len(host) > 253 || strings.ToLower(host) != host || strings.HasSuffix(host, ".") || strings.ContainsAny(host, " /\\@%?#\t\r\n") || seen[remote] {
			return ErrConfiguration
		}
		if net.ParseIP(host) == nil {
			for _, label := range strings.Split(host, ".") {
				if label == "" || len(label) > 63 || label[0] == '-' || label[len(label)-1] == '-' {
					return ErrConfiguration
				}
				for _, c := range label {
					if !(c >= 'a' && c <= 'z' || c >= '0' && c <= '9' || c == '-') {
						return ErrConfiguration
					}
				}
			}
		}
		seen[remote] = true
	}
	return nil
}
