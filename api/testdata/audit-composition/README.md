# Audit preparation contracts

`prepared-openapi-scan.yaml` is a catalog-valid, execution-gated profile using
existing Workflows. It is not installed as a runnable preset.

`profile-cases.json` drives `TestAuditPreparationAuthoringFixtures`: valid and
renamed roles, finite attempt bounds, invalid scopes, dependencies, obsolete
schema fields and executor/input mismatches. Additional Go tests cover two
prepare roles, deep copies and tampered persisted snapshots.

`public-cases.json` is shared by `TestPublicAuditPreparationContracts` and the
UI API parser tests. Both validate the same source mappings and retained output
shapes. Public tests also cover absent Round projections across pause/cancel/
delete/failure states; actual controller transitions belong to V62-003.

The single current schema rejects `sourceInput` and `settingsInput`. Profile
snapshot tests round-trip current canonical bytes; no old-schema reader exists.
