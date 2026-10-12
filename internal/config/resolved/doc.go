// Package resolved is the resolved Workflow model: the immutable,
// dependency-resolved Workflow that a Run stores as its snapshot, the Stage
// copy a StageExecution keeps, the Run's reference-only execution
// configuration patch, and every rule that checks such a value without the
// catalog (graph, execution configuration, scan plans and credentials). It
// also owns the one registry of code descriptors (PlannerFactory,
// WorkerRuntime, Toolset and SandboxProfile) that configuration may select.
//
// The catalog loader in internal/config resolves YAML into these values. The
// Scheduler, Run creation, Audits, Evals, Finding intake and placement read
// them back through the decoders and checks here, so they depend on this
// package and not on the loader. The JSON encoding of these types is the
// persisted snapshot format and the input of Eval pin digests.
package resolved
