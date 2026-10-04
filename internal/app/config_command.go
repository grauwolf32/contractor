package app

import (
	"flag"
	"fmt"
	"io"
	"log/slog"

	contractorconfig "github.com/grauwolf32/contractor/internal/config"
)

func runConfigCLI(args []string, logger *slog.Logger) error {
	if err := commandGroupHelp(args, "config",
		[2]string{"validate", "check the operator and managed configuration roots offline"}); err != nil {
		return err
	}
	if len(args) == 0 || args[0] != "validate" {
		return fmt.Errorf("config command requires the validate subcommand")
	}

	root := "./configs"
	managedRoot := ""
	flags := flag.NewFlagSet("contractor-server config validate", flag.ContinueOnError)
	flags.SetOutput(io.Discard)
	flags.StringVar(&root, "root", root, "configuration root")
	flags.StringVar(&managedRoot, "managed-root", managedRoot, "managed configuration root (default: sibling managed-configs)")
	if err := parseCommandFlags(flags, args[1:]); err != nil {
		return fmt.Errorf("parse config validate flags: %w", err)
	}
	if flags.NArg() != 0 {
		return fmt.Errorf("unexpected positional arguments: %v", flags.Args())
	}

	if managedRoot == "" {
		managedRoot = defaultManagedConfigRoot(root)
	}
	snapshot, err := contractorconfig.LoadUnionReadOnly(root, managedRoot, contractorconfig.MVPDescriptors())
	if err != nil {
		return fmt.Errorf("validate configuration: %w", err)
	}
	if err := validateBundledCatalogStandards(root, snapshot); err != nil {
		return fmt.Errorf("validate configuration: %w", err)
	}
	counts := snapshot.Counts()
	logger.Info(
		"configuration valid",
		"root", root,
		"managed_root", managedRoot,
		"workflows", counts.Workflows,
		"agent_templates", counts.AgentTemplates,
		"model_policies", counts.ModelPolicies,
		"llm_gateways", counts.LLMGateways,
		"execution_configs", counts.ExecutionConfigs,
		"audit_profiles", counts.AuditProfiles,
		"instructions", counts.Instructions,
	)
	return nil
}
