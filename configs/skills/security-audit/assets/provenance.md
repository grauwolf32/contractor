# Cloudflare security-audit import

Source: https://github.com/cloudflare/security-audit-skill/tree/c1c8a8c1471069fb0e188eeaff69b8e8db6564a8/skills/security-audit
Revision: `c1c8a8c1471069fb0e188eeaff69b8e8db6564a8`
Imported: 2026-10-22
License: MIT; the original copyright and license are in `assets/license.txt`.

All fourteen upstream Markdown companions are retained under lowercase
`references/` paths. The report schema is under `assets/`; the two production
validators are under `scripts/`. Upstream validator test programs are not part
of the model-facing package.

Changes: portable resource paths and links; native progressive-disclosure and
capability guidance; authorized workspace/artifact outputs instead of host paths;
explicitly incomplete full audits when delegation, verification, sandboxing or
validators are unavailable. The findings validator locates its schema at
`../assets/report-schema.json`; validator logic and the schema are unchanged.

Scripts are returned as text and are never executed by the Skill toolset. A Worker
may materialize resources through an available workspace writer and invoke them
only through its independently selected sandbox execution tool. No interpreter,
dependency installation, host access or additional tool is granted by this package.
