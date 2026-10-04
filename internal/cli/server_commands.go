package cli

import (
	"context"
	"errors"
	"fmt"

	"github.com/grauwolf32/contractor/internal/app"
)

func (c *CLI) runServer(ctx context.Context, args []string) error {
	if len(args) > 0 && helpArgument(args[0]) {
		_, _ = fmt.Fprintln(c.stderr, `Usage: contractor server <command> [flags]

Commands:
  run                 Run the Server
  migrate             Apply database migrations
  config validate     Validate the operator and managed configuration roots offline
  auth hash-password  Print a local-auth bootstrap document

Run 'contractor server <command> --help' for its flags.`)
		return nil
	}
	if len(args) == 0 {
		return &UsageError{Message: "server requires run, migrate, config validate, or auth hash-password"}
	}
	switch args[0] {
	case "run":
		return app.RunCLI(ctx, append([]string{"serve"}, args[1:]...), c.getenv, loggerTo(c.stderr))
	case "migrate":
		return app.RunCLI(ctx, append([]string{"migrate"}, args[1:]...), c.getenv, loggerTo(c.stderr))
	case "config":
		// Group help passes through so the Server prints the group's usage.
		if len(args) < 2 || args[1] != "validate" && !helpArgument(args[1]) {
			return &UsageError{Message: "usage: contractor server config validate [flags]"}
		}
		return app.RunCLI(ctx, append([]string{"config"}, args[1:]...), c.getenv, loggerTo(c.stderr))
	case "auth":
		if len(args) < 2 || args[1] != "hash-password" && !helpArgument(args[1]) {
			return &UsageError{Message: "usage: contractor server auth hash-password [flags]"}
		}
		return app.RunCLI(ctx, append([]string{"auth"}, args[1:]...), c.getenv, loggerTo(c.stderr))
	default:
		return errors.New("unknown server command " + args[0])
	}
}

// helpArgument reports the spellings the flag package treats as a help request.
func helpArgument(argument string) bool {
	switch argument {
	case "-h", "-help", "--h", "--help":
		return true
	}
	return false
}
