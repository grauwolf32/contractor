package app

import (
	"os"
	"path/filepath"
	"testing"
)

func TestGitConfigurationPrecedenceAndValidation(t *testing.T) {
	env := func(k string) string {
		switch k {
		case "CONTRACTOR_GIT_ALLOWED_REMOTES":
			return "git.example:22,github.com:443"
		case "CONTRACTOR_GIT_KNOWN_HOSTS_FILE":
			return "/config/known_hosts"
		}
		return ""
	}
	cfg, err := ParseConfig([]string{"--git-allowed-remote=gitlab.example:22"}, env)
	if err != nil || len(cfg.GitImport.AllowedRemotes) != 1 || cfg.GitImport.AllowedRemotes[0] != "gitlab.example:22" || cfg.GitImport.KnownHostsFile != "/config/known_hosts" {
		t.Fatalf("configuration: %+v %v", cfg.GitImport, err)
	}
	for _, value := range []string{"host", "host:0", "https://host:443", "*.host:22", "host:65536", "host:022", "host:22,host:22"} {
		if _, err := ParseConfig(nil, func(k string) string {
			if k == "CONTRACTOR_GIT_ALLOWED_REMOTES" {
				return value
			}
			return ""
		}); err == nil {
			t.Fatalf("accepted %q", value)
		}
	}
	if _, err := ParseConfig([]string{"--git-known-hosts-file=relative"}, func(string) string { return "" }); err == nil {
		t.Fatal("relative trust path accepted")
	}
	cfg, err = ParseConfig(nil, func(string) string { return "" })
	if err != nil || len(cfg.GitImport.AllowedRemotes) != 0 {
		t.Fatal("empty Git configuration affected Server defaults")
	}
}

func TestGitConfigurationYAMLIsOverriddenByEnvironmentAndCLI(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "server.yaml")
	if err := os.WriteFile(path, []byte("apiVersion: contractor/v1alpha1\nkind: ServerConfig\nspec:\n  gitAllowedRemotes: [git.example:22]\n  gitKnownHostsFile: known_hosts\n"), 0600); err != nil {
		t.Fatal(err)
	}
	cfg, err := ParseConfig([]string{"--config", path}, func(string) string { return "" })
	if err != nil || len(cfg.GitImport.AllowedRemotes) != 1 || cfg.GitImport.KnownHostsFile != filepath.Join(root, "known_hosts") {
		t.Fatalf("YAML settings: %+v %v", cfg.GitImport, err)
	}
	cfg, err = ParseConfig([]string{"--config", path, "--git-allowed-remote=cli.example:443"}, func(k string) string {
		if k == "CONTRACTOR_GIT_ALLOWED_REMOTES" {
			return "env.example:22"
		}
		return ""
	})
	if err != nil || len(cfg.GitImport.AllowedRemotes) != 1 || cfg.GitImport.AllowedRemotes[0] != "cli.example:443" {
		t.Fatalf("CLI precedence: %+v %v", cfg.GitImport, err)
	}
}
