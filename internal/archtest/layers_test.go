package archtest

// This file is the Server's layer table: the single place that says which
// way Go dependencies may point. TestLayerDependencies (check_test.go) reads
// the imports of every non-test package under api/, cmd/, internal/ and
// tools/ and enforces it:
//
//   - every package belongs to exactly one layer;
//   - a package imports only packages of its own layer or a lower one;
//   - an upward import fails unless knownViolations lists it with a reason
//     and the intended fix;
//   - a knownViolations entry that no longer matches an upward import fails,
//     so fixed violations leave the list.
//
// Go itself rejects import cycles, so imports within one layer are allowed.

// layers lists the layers from the bottom up. An entry is a package path
// relative to the module root; "dir/..." covers dir and every package below
// it, and an exact entry wins over a pattern.
var layers = []layerSpec{
	{
		name: "foundation",
		role: "dependency-free helpers, embedded API schemas and test support",
		packages: []string{
			"api/...",
			"internal/artifactpolicy",
			"internal/cabundle",
			"internal/clone",
			"internal/configtest",
			"internal/contentdigest",
			// Wire fixture helpers for contract tests; standard library only.
			"internal/contracts/contractstest",
			"internal/credentialerrors",
			"internal/documentnumber",
			"internal/httpapi/httpx",
			"internal/mtlstest",
			"internal/profiling",
			"internal/publicclient/generated",
			"internal/randomid",
			"internal/requestid",
			"internal/securefile",
			"internal/sourcezip",
			"internal/strictjson",
			"internal/zipdirectory",
		},
	},
	{
		name: "kernel",
		role: "wire contracts, persistence, authentication and transport security",
		packages: []string{
			"internal/auth",
			"internal/contracts",
			"internal/contracts/control",
			"internal/contracts/llmgateway",
			"internal/contracts/reporting",
			"internal/contracts/runlabels",
			"internal/contracts/runtimesettings",
			"internal/contracts/scan",
			"internal/localpki",
			"internal/mtls",
			"internal/persistence",
			"internal/persistence/migrations",
			"internal/persistence/postgres",
		},
	},
	{
		name: "platform",
		role: "infrastructure services: artifacts, telemetry, settings, previews",
		packages: []string{
			"internal/artifactpreview",
			"internal/artifacts",
			"internal/gatewayrecovery",
			"internal/httpapi/artifacttransfer",
			"internal/performance",
			"internal/settingsstore",
			"internal/sourcebundle",
			"internal/telemetry",
		},
	},
	{
		name: "catalog",
		role: "static catalogs and executable configuration",
		packages: []string{
			"internal/agentskills",
			"internal/auditstandards",
			"internal/config",
			// Loads catalogs together with the bundled skill and Audit standard checks.
			"internal/configload",
		},
	},
	{
		name: "domain",
		role: "pure domain cores without database access",
		packages: []string{
			"internal/auditdomain",
			"internal/auditpriority",
			"internal/evaldomain",
			"internal/scanplan",
		},
	},
	{
		name: "execution",
		role: "run execution: run store, runtime configuration, credentials, memory, planner, scheduler, control plane",
		packages: []string{
			"internal/controlplane",
			"internal/credentials",
			"internal/credentials/litellm",
			"internal/memory",
			"internal/planner",
			"internal/planner/a2a",
			"internal/planner/router",
			"internal/planner/session",
			"internal/planner/stateview",
			"internal/planner/streamline",
			"internal/runrepeat",
			"internal/runstore",
			"internal/runtimeconfig",
			"internal/scheduler",
		},
	},
	{
		name: "product",
		role: "product contexts and application services: audits, evals, projects, runs",
		packages: []string{
			"internal/auditbaseline",
			"internal/auditcontroller",
			"internal/auditimport",
			"internal/auditscan",
			"internal/auditservice",
			"internal/auditstore",
			"internal/evalcoordinator",
			"internal/evalservice",
			"internal/evalstore",
			"internal/findingintake",
			"internal/gitimport",
			// The planner adapter that runs audit scans.
			"internal/planner/scan",
			"internal/projectlifecycle",
			"internal/projectstore",
			"internal/runservice",
		},
	},
	{
		name: "edge",
		role: "HTTP adapters and the public API client",
		packages: []string{
			"internal/httpapi",
			"internal/httpapi/privateartifacts",
			"internal/httpapi/public",
			"internal/httpapi/public/events",
			"internal/publicclient",
		},
	},
	{
		name: "composition",
		role: "process wiring and commands",
		packages: []string{
			"cmd/...",
			"internal/app",
			"internal/cli",
			"tools/...",
		},
	},
}

// knownViolations are the upward imports that exist today. Each entry names
// why the import exists and how it should be removed. Do not add an entry to
// silence a new violation without both.
var knownViolations = []knownViolation{
	{
		from:   "internal/runstore",
		to:     "internal/auditstore",
		reason: "delete_store.go invalidates Audit projections of a deleted WorkflowRun inside the delete transaction",
		fix:    "invert it: runstore takes a run-deletion hook that runs in the transaction, and internal/app wires auditstore into it",
	},
}
