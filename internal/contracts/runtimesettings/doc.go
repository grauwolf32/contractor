// Package runtimesettings holds the Runtime settings delivered with an
// allocation (LLM Gateway access, telemetry export, HTTP proxy, Caido and the
// Project HTTP origin target), the non-secret refs a Project or Run pins for
// them, and the resolved RuntimeConfig provenance they are selected by.
//
// Settings carry SecretString values: they are private-wire values that must
// never be logged or persisted in plaintext.
package runtimesettings
