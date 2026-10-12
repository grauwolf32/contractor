// Package agentprincipal is the durable registry of Runtime Agent
// principals: the certificate-key identity a Runtime Agent registers with,
// the Runtime labels assigned to it and their revision. It seeds a principal
// at its first registration, replaces labels and deletes principals under
// CAS with idempotent Runtime management receipts, and serves RuntimeConfig
// label management the principals that hold a label
// (runtimeconfig.LabelHolders). The label grammar, bindings and their
// applicability to a Runtime Agent stay with internal/runtimeconfig.
//
// The registry reports its outcomes with the RuntimeConfig error classes
// every consumer already maps: a missing principal is runtimeconfig.ErrNotFound,
// a stale revision runtimeconfig.ErrPrecondition, an invalid request
// runtimeconfig.ErrInvalid and an unknown label runtimeconfig.ErrUnknownLabel.
package agentprincipal
