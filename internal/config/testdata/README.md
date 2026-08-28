# Configuration loader fixtures

`valid/` is a complete, dependency-resolved configuration root. Files in
`invalid/` are focused replacements for the valid ModelPolicy manifest; tests
install one replacement at a time so every failure remains attributable to one
contract violation.
