package app

import (
	"flag"
	"fmt"
	"io"
)

func (c *serveConfigInputs) parseFlags(args []string) error {
	flags := flag.NewFlagSet("contractor-server serve", flag.ContinueOnError)
	flags.SetOutput(io.Discard)
	c.settings.operations.registerFlags(flags)
	flags.Var(&c.gitRemotes, "git-allowed-remote", "allowed Git remote host:port; repeatable")
	flags.StringVar(&c.gitKnownHosts, "git-known-hosts-file", c.gitKnownHosts, "absolute read-only SSH known_hosts file")
	flags.StringVar(&c.serverConfigPath, "config", c.serverConfigPath, "strict YAML ServerConfig file")
	flags.StringVar(&c.serverConfigPath, "server-config", c.serverConfigPath, "alias for --config")
	flags.StringVar(&c.blobBackend, "artifact-blob-backend", c.blobBackend, "Artifact payload backend: postgresql or filesystem")
	flags.StringVar(&c.blobPath, "artifact-blob-path", c.blobPath, "absolute filesystem blob directory")
	flags.Var(&c.performanceMetrics, "performance-metrics", "enable Operations performance collection (requires restart)")
	flags.Var(&c.pprof, "pprof", "enable independent loopback Go profiling (requires restart)")
	flags.StringVar(&c.pprofListen, "pprof-listen", c.pprofListen, "numeric loopback Go profiling listen address")
	flags.StringVar(&c.listenAddress, "listen", c.listenAddress, "public HTTP listen address")
	flags.StringVar(&c.privateListenAddress, "private-listen", c.privateListenAddress, "private mTLS listen address")
	flags.StringVar(&c.privateURL, "private-url", c.privateURL, "advertised private mTLS base URL")
	flags.DurationVar(
		&c.shutdownTimeout,
		"shutdown-timeout",
		c.shutdownTimeout,
		"graceful shutdown timeout",
	)
	flags.DurationVar(
		&c.runtimeRequestTimeout,
		"runtime-request-timeout",
		c.runtimeRequestTimeout,
		"private Runtime Agent request timeout",
	)
	flags.DurationVar(
		&c.workerRequestTimeout,
		"worker-request-timeout",
		c.workerRequestTimeout,
		"allocation-local Worker outbound request timeout (minimum 120s)",
	)
	flags.StringVar(&c.databaseURL, "database-url", c.databaseURL, "PostgreSQL connection URL")
	flags.StringVar(&c.operatorConfigRoot, "operator-config-root", c.operatorConfigRoot, "operator/bootstrap configuration root")
	flags.StringVar(&c.managedConfigRoot, "managed-config-root", c.managedConfigRoot, "Server-managed configuration publication root")
	flags.StringVar(
		&c.credentialMasterKeyFile,
		"credential-master-key-file",
		c.credentialMasterKeyFile,
		"absolute owner-only file containing the credential encryption key",
	)
	flags.StringVar(
		&c.llmGatewayAdminBindingsFile,
		"llm-gateway-admin-bindings-file",
		c.llmGatewayAdminBindingsFile,
		"absolute strict YAML file binding exact LLM Gateways to owner-only admin-key files",
	)
	flags.StringVar(&c.localAuthFile, "local-auth-file", c.localAuthFile, "absolute owner-only local authentication YAML")
	flags.Var(&c.browserOrigins, "browser-origin", "exact allowed browser UI origin; repeat for multiple origins")
	flags.BoolVar(
		&c.insecureLoopbackCookie,
		"insecure-loopback-cookie",
		c.insecureLoopbackCookie,
		"use the separately named insecure cookie on an IP-literal loopback listener",
	)
	flags.StringVar(&c.caFile, "ca-file", c.caFile, "deployment CA certificate")
	flags.StringVar(&c.certificateFile, "certificate-file", c.certificateFile, "Control Plane certificate")
	flags.StringVar(&c.privateKeyFile, "private-key-file", c.privateKeyFile, "Control Plane private key")
	flags.DurationVar(&c.plannerTimeout, "planner-timeout", c.plannerTimeout, "maximum Stage preparation and Planner wall time")
	flags.StringVar(&c.developmentLLMGateway, "development-llm-gateway", c.developmentLLMGateway, "exact LLM Gateway selector for development token bindings")
	if err := flags.Parse(args); err != nil {
		return fmt.Errorf("parse serve flags: %w", err)
	}
	if flags.NArg() != 0 {
		return fmt.Errorf("unexpected positional arguments: %v", flags.Args())
	}
	return nil
}
