package app

import (
	"flag"
	"fmt"
	"io"
	"log/slog"

	contractorconfig "github.com/grauwolf32/contractor/internal/config"
)

func runConfigCLI(args []string, logger *slog.Logger) error {
	if len(args) == 0 || args[0] != "validate" {
		return fmt.Errorf("config command requires the validate subcommand")
	}

	root := "./configs"
	flags := flag.NewFlagSet("contractor-server config validate", flag.ContinueOnError)
	flags.SetOutput(io.Discard)
	flags.StringVar(&root, "root", root, "configuration root")
	if err := flags.Parse(args[1:]); err != nil {
		return fmt.Errorf("parse config validate flags: %w", err)
	}
	if flags.NArg() != 0 {
		return fmt.Errorf("unexpected positional arguments: %v", flags.Args())
	}

	snapshot, err := contractorconfig.Load(root, contractorconfig.MVPDescriptors())
	if err != nil {
		return fmt.Errorf("validate configuration: %w", err)
	}
	counts := snapshot.Counts()
	logger.Info(
		"configuration valid",
		"root", root,
		"workflows", counts.Workflows,
		"agent_templates", counts.AgentTemplates,
		"model_policies", counts.ModelPolicies,
		"llm_gateways", counts.LLMGateways,
		"execution_configs", counts.ExecutionConfigs,
		"instructions", counts.Instructions,
	)
	return nil
}
