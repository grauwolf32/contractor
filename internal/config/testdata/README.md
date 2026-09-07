# Configuration loader fixtures

`valid/` is a complete, dependency-resolved configuration root. Files in
`invalid/` are focused replacements for the valid ModelPolicy manifest; tests
install one replacement at a time so every failure remains attributable to one
contract violation. The valid root also carries an unconsumed ExecutionConfig
profile so every required immutable manifest subtree is represented.

Shared policy-override and escalation fixtures live in
[`../../configtest/testdata/`](../../configtest/testdata/README.md). They use
`test-*` policy names and are installed only into temporary test catalogs.
