---
description: Vulnerability scanning skill — workflow overview, checklist references, and common miss patterns. Read this first, then load references on demand with skills_read.
---

# Vulnerability Scanning Skill

## Workflow

1. **Inventory** — `ls /` → `attack_surface` → identify framework, language, entry points
2. **Read handlers** — read each handler/route file; for each endpoint, fill the control checklist
3. **Read models + config** — read data models, config files, init/seed scripts, requirements
4. **Report** — `report_vulnerability` for each confirmed finding

## References (load on demand)

- `vuln_scan/references/checklist` — per-endpoint control checklist (auth, authz, ownership, validation, output filtering, rate limiting)
- `vuln_scan/references/miss-patterns` — commonly missed vulnerability patterns with examples
