# Bundled UI widgets

Widgets own specialized presentation. Their hosts own authenticated APIs,
query/poll lifecycles, creation keys, navigation and authorized evidence links.
The shared `src/ui` building blocks still own controls, focus and themes.

- `src/workflow-widgets`: exact Workflow contract dispatch and Nuclei launch
  controls. Audit scan results render the registered
  React fields. Compatibility checks cover version, parameters and input scope.
- `scan-reports`: versioned scanner result presentation, dispatched by the
  report's tool. Nuclei, Naabu and SQLMap use the shared outcome view; ffuf
  requires complete scan accounting, and Katana adds bounded discovery facts.
  Unknown reports retain the ordinary raw JSON preview. The error outcome table
  is shared across these adapters and tested against the wire fixture.
- `audit-scans`: Nuclei URL and SQLMap request presentation plus their coverage
  kind identity. The Audit host supplies source/result links and their existing
  receipts. Unsupported scanners retain explicit evidence access without being
  interpreted as SQLMap.

Register only widgets with a production caller. Keep ordinary Run lifecycle,
Project selection, artifact transport and MIME-based OpenAPI/LikeC4 previews
shared. Widgets neither execute Workflows nor change their Server contracts.
No dynamic code loading, user-defined JavaScript or Studio editor is involved.
