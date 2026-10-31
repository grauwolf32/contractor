# Deployment tooling

This directory contains reusable scripts, pinned dependencies and illustrative
examples. Keep concrete installation settings and operational state under the
Git-ignored `.local/` directory:

| Versioned in `deploy/` | Private in `.local/` |
| --- | --- |
| Deployment scripts and Ansible roles | SSH aliases, inventory, actual addresses and network selections |
| Image/tool version pins and shared model catalog | Credentials, PKI, VPN profiles and keys |
| `*.example.*` configuration and generic instructions | Server profiles, launch environment, reports, backups and local runbooks |

Copy examples into `.local/` and substitute your installation's values there.
Do not edit reusable scripts to set an address, a provider mirror or a password.
The GLM/OpenRouter profiles in [models](models/) are a shared catalog; this
separation does not change model selections or generated policy identities.

- [Ansible Runtime provisioning](ansible/README.md)
- [Hybrid controller, Runtime, Gateway and WireGuard tooling](hybrid/README.md)
- [LiteLLM local launcher](../docs/deployment.md#llm-gateway)
- [UI container](ui/README.md)
- [Podman execution](podman/README.md)
- [Phoenix trace viewer](phoenix/README.md)
- [Complete deployment guide](../docs/deployment.md)

The hybrid helpers use `.local/hybrid/deployment.json` as their operator input.
All accept an explicit `--state` directory, which is useful from a worktree.
No installation addresses are inferred from the examples. Follow the component
guide's teardown, backup and restart requirements when changing live inputs.
