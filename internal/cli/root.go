package cli

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log/slog"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/grauwolf32/contractor/internal/publicclient"
)

var Version = "dev"

type Environment func(string) string

type CLI struct {
	stdin  io.Reader
	stdout io.Writer
	stderr io.Writer
	getenv Environment
}

func New(stdin io.Reader, stdout, stderr io.Writer, getenv Environment) *CLI {
	if getenv == nil {
		getenv = os.Getenv
	}
	return &CLI{stdin: stdin, stdout: stdout, stderr: stderr, getenv: getenv}
}

type UsageError struct{ Message string }

func (e *UsageError) Error() string { return e.Message }

func (c *CLI) Run(ctx context.Context, args []string) error {
	options, command, commandArgs, err := c.parseGlobal(args)
	if err != nil {
		return err
	}
	printer, err := NewPrinter(options.output, c.stdout)
	if err != nil {
		return &UsageError{Message: err.Error()}
	}
	switch command {
	case "help-shown":
		return nil
	case "help":
		c.writeRootUsage()
		return nil
	case "version":
		return printer.Names(Version)
	case "server":
		return c.runServer(ctx, commandArgs)
	case "pki":
		return c.runPKI(commandArgs, printer)
	}
	if command != "context" && command != "contexts" && command != "workflow" && command != "workflows" &&
		command != "project" && command != "projects" && command != "artifact" && command != "artifacts" &&
		command != "source" && command != "run" && command != "runs" && command != "queue" &&
		command != "ops" && command != "operations" && command != "check" {
		return &UsageError{Message: fmt.Sprintf("unknown command %q", command)}
	}

	configPath, err := DefaultContextPath(c.getenv)
	if err != nil {
		return err
	}
	store := NewContextStore(configPath)
	if command == "context" || command == "contexts" {
		return c.runContext(ctx, store, printer, commandArgs)
	}

	client, selected, err := c.client(options, store)
	if err != nil {
		return err
	}
	switch command {
	case "workflow", "workflows":
		return c.runWorkflow(ctx, client, printer, commandArgs)
	case "project", "projects":
		return c.runProject(ctx, client, printer, commandArgs)
	case "artifact", "artifacts":
		return c.runArtifact(ctx, client, printer, commandArgs)
	case "source":
		return c.runSource(ctx, client, printer, commandArgs)
	case "run", "runs":
		return c.runRun(ctx, client, printer, commandArgs)
	case "queue":
		return c.runQueue(ctx, client, printer, commandArgs)
	case "ops", "operations":
		return c.runOperations(ctx, client, printer, commandArgs)
	case "check":
		return c.checkContext(ctx, client, printer, selected)
	default:
		return &UsageError{Message: fmt.Sprintf("unknown command %q", command)}
	}
}

type globalOptions struct {
	contextName string
	server      string
	caFile      string
	tokenFile   string
	allowHTTP   bool
	output      string
	timeout     time.Duration
	caFileSet   bool
	tokenSet    bool
	httpSet     bool
}

func (c *CLI) parseGlobal(args []string) (globalOptions, string, []string, error) {
	options := globalOptions{
		contextName: c.getenv("CONTRACTOR_CONTEXT"),
		server:      c.getenv("CONTRACTOR_SERVER"),
		caFile:      c.getenv("CONTRACTOR_API_CA_FILE"),
		tokenFile:   c.getenv("CONTRACTOR_API_TOKEN_FILE"),
		output:      c.getenv("CONTRACTOR_OUTPUT"),
		timeout:     30 * time.Second,
	}
	if options.output == "" {
		options.output = string(OutputTable)
	}
	if encoded := c.getenv("CONTRACTOR_TIMEOUT"); encoded != "" {
		parsed, err := time.ParseDuration(encoded)
		if err != nil {
			return globalOptions{}, "", nil, &UsageError{Message: "invalid CONTRACTOR_TIMEOUT"}
		}
		options.timeout = parsed
	}
	if encoded := c.getenv("CONTRACTOR_ALLOW_HTTP"); encoded != "" {
		parsed, err := strconv.ParseBool(encoded)
		if err != nil {
			return globalOptions{}, "", nil, &UsageError{Message: "CONTRACTOR_ALLOW_HTTP must be true or false"}
		}
		options.allowHTTP = parsed
		options.httpSet = true
	}
	flags := flag.NewFlagSet("contractor", flag.ContinueOnError)
	flags.SetOutput(c.stderr)
	flags.StringVar(&options.contextName, "context", options.contextName, "named Server context")
	flags.StringVar(&options.server, "server", options.server, "Contractor Server origin")
	flags.StringVar(&options.caFile, "ca-file", options.caFile, "additional public API CA bundle")
	flags.StringVar(&options.tokenFile, "token-file", options.tokenFile, "file containing the bearer token")
	flags.BoolVar(&options.allowHTTP, "allow-http", options.allowHTTP, "allow cleartext HTTP to a non-loopback Server")
	flags.StringVar(&options.output, "output", options.output, "output format: table, json, or name")
	flags.DurationVar(&options.timeout, "timeout", options.timeout, "per-request timeout")
	flags.Usage = c.writeRootUsage
	if err := flags.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return options, "help-shown", nil, nil
		}
		return globalOptions{}, "", nil, &UsageError{Message: err.Error()}
	}
	flags.Visit(func(current *flag.Flag) {
		switch current.Name {
		case "ca-file":
			options.caFileSet = true
		case "token-file":
			options.tokenSet = true
		case "allow-http":
			options.httpSet = true
		}
	})
	remaining := flags.Args()
	if len(remaining) == 0 {
		return globalOptions{}, "", nil, &UsageError{Message: "a command is required"}
	}
	return options, remaining[0], remaining[1:], nil
}

func (c *CLI) client(options globalOptions, store *ContextStore) (*publicclient.Client, string, error) {
	selected := "direct"
	if options.contextName != "" || options.server == "" {
		config, err := store.Load()
		if err != nil {
			return nil, "", err
		}
		name, context, resolveErr := config.Resolve(options.contextName)
		if resolveErr != nil {
			return nil, "", resolveErr
		}
		selected = name
		if options.server == "" {
			options.server = context.Server
		}
		if options.caFile == "" && !options.caFileSet {
			options.caFile = context.CAFile
		}
		if options.tokenFile == "" && !options.tokenSet {
			options.tokenFile = context.TokenFile
		}
		if !options.httpSet {
			options.allowHTTP = context.AllowHTTP
		}
	}
	var err error
	token := c.getenv("CONTRACTOR_API_TOKEN")
	if token == "" && options.tokenFile != "" {
		token, err = publicclient.ReadTokenFile(options.tokenFile)
		if err != nil {
			return nil, "", err
		}
	}
	if token == "" {
		return nil, "", errors.New("API token is required; set CONTRACTOR_API_TOKEN or --token-file")
	}
	client, err := publicclient.New(publicclient.Options{
		Server: options.server, Token: token, CAFile: options.caFile,
		AllowHTTP: options.allowHTTP, Timeout: options.timeout,
		UserAgent: "contractor-cli/" + Version,
	})
	return client, selected, err
}

func (c *CLI) writeRootUsage() {
	_, _ = fmt.Fprintln(c.stderr, `Usage: contractor [global flags] <command> [args]

Commands:
  context     Manage named Server contexts
  check       Verify connectivity, authentication, and API compatibility
  workflow    List and inspect published Workflows
  project     Manage Projects and Project-scoped resources
  artifact    Manage User, Project, and Run Artifacts
  source      Package and upload a source directory
  run         Create, inspect, watch, cancel, and download Run outputs
  queue       Inspect, pause, and resume Run admission
  ops         Inspect Runtime Agents and allocations
  pki         Generate a local CA and Runtime mTLS certificates
  server      Run existing Server administration commands
  version     Print CLI version

Global flags must appear before the command. Authentication uses
CONTRACTOR_API_TOKEN or --token-file.`)
}

func parseFlags(name string, output io.Writer, args []string, configure func(*flag.FlagSet)) (*flag.FlagSet, error) {
	flags := flag.NewFlagSet(name, flag.ContinueOnError)
	flags.SetOutput(output)
	configure(flags)
	if err := flags.Parse(interspersedFlags(flags, args)); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return nil, nil
		}
		return nil, &UsageError{Message: err.Error()}
	}
	return flags, nil
}

// The standard flag package stops at the first positional argument. CLI users
// conventionally put command flags before or after resource names, so move
// recognized flags ahead of positional arguments while preserving both groups.
func interspersedFlags(flags *flag.FlagSet, args []string) []string {
	var options, positionals []string
	for index := 0; index < len(args); index++ {
		argument := args[index]
		if argument == "--" {
			positionals = append(positionals, args[index+1:]...)
			break
		}
		if argument == "-" || !strings.HasPrefix(argument, "-") {
			positionals = append(positionals, argument)
			continue
		}
		name := strings.TrimLeft(argument, "-")
		if separator := strings.IndexByte(name, '='); separator >= 0 {
			name = name[:separator]
		}
		definition := flags.Lookup(name)
		options = append(options, argument)
		if definition == nil || strings.Contains(argument, "=") {
			continue
		}
		if boolean, ok := definition.Value.(interface{ IsBoolFlag() bool }); ok && boolean.IsBoolFlag() {
			continue
		}
		if index+1 < len(args) {
			index++
			options = append(options, args[index])
		}
	}
	return append(options, positionals...)
}

func requirePositionals(flags *flag.FlagSet, minimum, maximum int, usage string) ([]string, error) {
	if flags == nil {
		return nil, nil
	}
	values := flags.Args()
	if len(values) < minimum || len(values) > maximum {
		return nil, &UsageError{Message: usage}
	}
	return values, nil
}

func exactSelector(value string) (string, string, error) {
	name, version, found := strings.Cut(value, "@")
	if !found || name == "" || version == "" || strings.Contains(version, "@") {
		return "", "", &UsageError{Message: "selector must be name@version"}
	}
	return name, version, nil
}

func idempotencyKey(explicit string) (string, error) {
	if explicit != "" {
		return explicit, nil
	}
	return publicclient.NewIdempotencyKey()
}

func stringValue(value any) string {
	if value == nil {
		return ""
	}
	encoded, err := json.Marshal(value)
	if err == nil {
		var result string
		if json.Unmarshal(encoded, &result) == nil {
			return result
		}
	}
	return fmt.Sprint(value)
}

func loggerTo(writer io.Writer) *slog.Logger {
	return slog.New(slog.NewJSONHandler(writer, nil))
}
