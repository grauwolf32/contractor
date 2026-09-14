# Operations

[Documentation index](../README.md)

Start with [deployment](../deployment.md) for installation, topology and startup.
The references below cover ongoing operation and component-specific settings.

| Task | Guide |
| --- | --- |
| Configure process settings and independent deadlines | [Timeout configuration](timeout-configuration.md), [local ServerConfig](../../configs/server.local.yaml) |
| Inspect Runtime capabilities, bind labels and plan upgrades | [Runtime configuration](runtime-configuration.md), [deployment examples](../../deploy/runtime-labels/README.md) |
| Manage encrypted LiteLLM virtual keys | [Gateway credentials](gateway-credentials.md) |
| Choose PostgreSQL or filesystem payload storage and clean orphan blobs | [Artifact blob storage](artifact-blob-storage.md) |
| Configure Git SSH keys, import and retention | [Git artifacts](../guides/git-artifacts.md) |
| Deploy the independent Node UI | [UI deployment](../../deploy/ui/README.md) |
| Provision allocation-scoped Podman execution | [Podman deployment](../../deploy/podman/README.md), [Runtime policy](../../runtime/PODMAN.md) |
| Inspect metrics, retention and missing data | [Operations performance](performance.md) |
| Enable and use loopback Go profiling | [Go profiling](go-profiling.md) |
| Verify a deployment change | [Testing](../testing/README.md) |

Server, Runtime and UI settings have separate owners. Workflow execution
policies live in the [configuration catalog](../../configs/README.md);
Runtime labels select infrastructure; Run metadata labels group Runs.
See the [specification index](../spec/README.md) for those contracts.
