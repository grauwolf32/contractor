---
name: pseudo-skill
description: Test-only Agent Skill that exercises packaging, manifest metadata and resource resolution.
license: Test fixture only
compatibility: Contractor agentskills package tests
metadata:
  source-revision: 0123456789abcdef0123456789abcdef01234567
  fixture: pseudo
---
# Pseudo Skill

This document is test data, not a deployable Agent Skill.

## References

- `references/checklist.md`: the steps to follow.
- `references/patterns/matching.md`: nested reference material.

Load a reference with
load_skill_resource(skill_name="pseudo-skill", file_path="references/checklist.md").

The result layout lives in `assets/template.txt`.
