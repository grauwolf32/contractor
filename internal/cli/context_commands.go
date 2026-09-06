package cli

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"net/http"
	"time"

	"github.com/grauwolf32/contractor/internal/publicclient"
	publicapi "github.com/grauwolf32/contractor/internal/publicclient/generated"
)

func (c *CLI) runContext(ctx context.Context, store *ContextStore, printer *Printer, args []string) error {
	if len(args) == 0 {
		return &UsageError{Message: "context requires add, list, show, use, remove, or check"}
	}
	switch args[0] {
	case "add":
		var server, caFile, tokenFile string
		var allowHTTP, makeCurrent bool
		flags, err := parseFlags("contractor context add", c.stderr, args[1:], func(flags *flag.FlagSet) {
			flags.StringVar(&server, "server", "", "Contractor Server origin")
			flags.StringVar(&caFile, "ca-file", "", "additional public API CA bundle")
			flags.StringVar(&tokenFile, "token-file", "", "file containing the bearer token")
			flags.BoolVar(&allowHTTP, "allow-http", false, "allow cleartext non-loopback HTTP")
			flags.BoolVar(&makeCurrent, "use", false, "select the context after adding it")
		})
		if err != nil || flags == nil {
			return err
		}
		positionals, err := requirePositionals(flags, 1, 1, "usage: contractor context add <name> --server <origin>")
		if err != nil {
			return err
		}
		if server == "" {
			return &UsageError{Message: "context add requires --server"}
		}
		config, err := store.Put(positionals[0], ServerContext{
			Server: server, CAFile: caFile, TokenFile: tokenFile, AllowHTTP: allowHTTP,
		}, makeCurrent)
		if err != nil {
			return err
		}
		return printer.Object(config.Contexts[positionals[0]], positionals[0])
	case "list":
		if len(args) != 1 {
			return &UsageError{Message: "usage: contractor context list"}
		}
		config, err := store.Load()
		if err != nil {
			return err
		}
		if printer.Mode() == OutputJSON {
			return printer.JSON(config)
		}
		if printer.Mode() == OutputName {
			return printer.Names(config.Names()...)
		}
		rows := make([][]string, 0, len(config.Contexts))
		for _, name := range config.Names() {
			marker := ""
			if name == config.CurrentContext {
				marker = "*"
			}
			rows = append(rows, []string{marker, name, config.Contexts[name].Server})
		}
		return printer.Table([]string{"CURRENT", "NAME", "SERVER"}, rows)
	case "show":
		flags, err := parseFlags("contractor context show", c.stderr, args[1:], func(*flag.FlagSet) {})
		if err != nil || flags == nil {
			return err
		}
		positionals, err := requirePositionals(flags, 0, 1, "usage: contractor context show [name]")
		if err != nil {
			return err
		}
		config, err := store.Load()
		if err != nil {
			return err
		}
		name := ""
		if len(positionals) == 1 {
			name = positionals[0]
		}
		name, selected, err := config.Resolve(name)
		if err != nil {
			return err
		}
		return printer.Object(selected, name)
	case "use", "remove":
		if len(args) != 2 {
			return &UsageError{Message: "usage: contractor context " + args[0] + " <name>"}
		}
		var config ContextConfig
		var err error
		if args[0] == "use" {
			config, err = store.Use(args[1])
		} else {
			config, err = store.Remove(args[1])
		}
		if err != nil {
			return err
		}
		if printer.Mode() == OutputJSON {
			return printer.JSON(config)
		}
		return printer.Names(args[1])
	case "check":
		config, err := store.Load()
		if err != nil {
			return err
		}
		name := ""
		if len(args) == 2 {
			name = args[1]
		} else if len(args) != 1 {
			return &UsageError{Message: "usage: contractor context check [name]"}
		}
		name, selected, err := config.Resolve(name)
		if err != nil {
			return err
		}
		token := c.getenv("CONTRACTOR_API_TOKEN")
		if token == "" && selected.TokenFile != "" {
			token, err = publicclient.ReadTokenFile(selected.TokenFile)
		}
		if err != nil || token == "" {
			if err != nil {
				return err
			}
			return errors.New("API token is required; set CONTRACTOR_API_TOKEN or configure --token-file")
		}
		client, err := publicclient.New(publicclient.Options{
			Server: selected.Server, Token: token, CAFile: selected.CAFile,
			AllowHTTP: selected.AllowHTTP, Timeout: 30 * time.Second,
			UserAgent: "contractor-cli/" + Version,
		})
		if err != nil {
			return err
		}
		return c.checkContext(ctx, client, printer, name)
	default:
		return &UsageError{Message: fmt.Sprintf("unknown context command %q", args[0])}
	}
}

func (c *CLI) checkContext(ctx context.Context, client *publicclient.Client, printer *Printer, name string) error {
	limit := publicapi.Limit(1)
	response, err := client.API.ListWorkflowsWithResponse(ctx, &publicapi.ListWorkflowsParams{Limit: &limit})
	if err != nil {
		return err
	}
	if err := publicclient.CheckResponse(response, http.StatusOK); err != nil {
		return err
	}
	result := map[string]any{
		"context": name, "server": client.Origin(), "apiVersion": publicclient.APIVersion,
		"authenticated": true,
	}
	if printer.Mode() == OutputTable {
		return printer.Table([]string{"CONTEXT", "SERVER", "API", "AUTH"}, [][]string{{name, client.Origin(), publicclient.APIVersion, "ok"}})
	}
	return printer.Object(result, name)
}
