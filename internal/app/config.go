package app

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"strings"
	"time"
)

const (
	defaultListenAddress   = "127.0.0.1:8080"
	defaultShutdownTimeout = 5 * time.Second
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
	shutdownTimeout := defaultShutdownTimeout
	databaseURL := getenv("CONTRACTOR_DATABASE_URL")

	flags := flag.NewFlagSet("contractor-server serve", flag.ContinueOnError)
	flags.SetOutput(io.Discard)
	flags.StringVar(&listenAddress, "listen", listenAddress, "public HTTP listen address")
	flags.DurationVar(
		&shutdownTimeout,
		"shutdown-timeout",
		shutdownTimeout,
		"graceful shutdown timeout",
	)
	flags.StringVar(&databaseURL, "database-url", databaseURL, "PostgreSQL connection URL")
	if err := flags.Parse(args); err != nil {
		return Config{}, fmt.Errorf("parse serve flags: %w", err)
	}
	if flags.NArg() != 0 {
		return Config{}, fmt.Errorf("unexpected positional arguments: %v", flags.Args())
	}
	if listenAddress == "" {
		return Config{}, errors.New("listen address must not be empty")
	}
	if shutdownTimeout <= 0 {
		return Config{}, errors.New("shutdown timeout must be positive")
	}

	return Config{
		ListenAddress: listenAddress, ShutdownTimeout: shutdownTimeout, DatabaseURL: databaseURL,
	}, nil
}
