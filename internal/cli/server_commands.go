package cli

import (
	"context"
	"errors"

	"github.com/grauwolf32/contractor/internal/app"
)

func (c *CLI) runServer(ctx context.Context, args []string) error {
	if len(args) == 0 {
		return &UsageError{Message: "server requires run, migrate, config validate, or auth hash-password"}
	}
	switch args[0] {
	case "run":
		return app.RunCLI(ctx, append([]string{"serve"}, args[1:]...), c.getenv, loggerTo(c.stderr))
	case "migrate":
		return app.RunCLI(ctx, append([]string{"migrate"}, args[1:]...), c.getenv, loggerTo(c.stderr))
	case "config":
		if len(args) < 2 || args[1] != "validate" {
			return &UsageError{Message: "usage: contractor server config validate [flags]"}
		}
		return app.RunCLI(ctx, append([]string{"config", "validate"}, args[2:]...), c.getenv, loggerTo(c.stderr))
	case "auth":
		if len(args) < 2 || args[1] != "hash-password" {
			return &UsageError{Message: "usage: contractor server auth hash-password [flags]"}
		}
		return app.RunCLI(ctx, append([]string{"auth", "hash-password"}, args[2:]...), c.getenv, loggerTo(c.stderr))
	default:
		return errors.New("unknown server command " + args[0])
	}
}
