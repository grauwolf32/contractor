// Package secretholders is the fixture of the secret-holder guard
// (internal/archtest/secret_holders_test.go). Each type is either a finding
// the guard must report or a boundary it must accept; the comment says which.
package secretholders

import (
	"crypto/tls"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"sync"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/pentestcontract"
)

// Finding: an unexported SecretString field and no Format, so fmt prints the
// plaintext for every verb.
type unexportedSecret struct {
	name  string
	token contracts.SecretString
}

// Accepted: fmt calls SecretString's own Format through an exported field.
type ExportedSecret struct {
	Token contracts.SecretString
}

// Finding: a secret nested in a map value and a slice under an unexported field.
type nestedInMap struct {
	values map[string][]contracts.SecretString
}

// Finding: an exported field whose type keeps its secret in an unexported field.
type ExportedHolder struct {
	Inner unexportedSecret
}

// Finding: an embedded unexported type is an unexported field.
type embedsHolder struct {
	unexportedSecret
}

// Accepted: a holder that redacts itself as a whole.
type RedactingHolder struct {
	token contracts.SecretString
}

func (h RedactingHolder) String() string                 { return "RedactingHolder([REDACTED])" }
func (h RedactingHolder) GoString() string               { return h.String() }
func (h RedactingHolder) Format(state fmt.State, _ rune) { _, _ = io.WriteString(state, h.String()) }
func (h RedactingHolder) MarshalJSON() ([]byte, error)   { return json.Marshal(h.String()) }
func (h RedactingHolder) LogValue() slog.Value           { return slog.StringValue(h.String()) }

// Finding: fmt cannot call RedactingHolder's methods through an unexported
// field, and %t follows one pointer.
type onePointer struct {
	inner *RedactingHolder
}

// Accepted: fmt prints a second nested pointer as an address for every verb.
type twoPointers struct {
	next *onePointer
}

// Accepted: the guard cannot see the dynamic type behind an interface.
type behindInterface struct {
	value any
}

// Accepted: fmt calls the holder's own methods through exported fields.
type ExportedRedacting struct {
	Holder  RedactingHolder
	Pointer *RedactingHolder
}

// Finding: an unexported plain-text field named like a secret.
type plainPassword struct {
	password string
}

// Finding: an exported plain-text secret that is not a wire field.
type ExportedPlain struct {
	Password string
}

// Accepted: a JSON wire field carries the plaintext by contract.
type WireRequest struct {
	Password string `json:"password"`
}

// Accepted: fields that name, locate or digest a secret.
type secretReferences struct {
	tokenID      string
	passwordFile string
	csrfDigest   [32]byte
}

// Finding: the String/GoString-only pattern, which %d and other verbs bypass.
type stringOnlySecret struct {
	value string
}

func (stringOnlySecret) String() string   { return "[REDACTED]" }
func (stringOnlySecret) GoString() string { return "stringOnlySecret([REDACTED])" }

// Finding: a held secret type, whatever its own methods.
type heldStringOnly struct {
	secret stringOnlySecret
}

// Finding: Format on the pointer does not cover a copied value.
type pointerFormatted struct {
	token contracts.SecretString
}

func (h *pointerFormatted) Format(state fmt.State, _ rune) {
	_, _ = io.WriteString(state, "[REDACTED]")
}

// Accepted: a type holding a lock may redact through its pointer only.
type lockedHolder struct {
	mu    sync.Mutex
	token contracts.SecretString
}

func (h *lockedHolder) Format(state fmt.State, _ rune) { _, _ = io.WriteString(state, "[REDACTED]") }

// Finding: listed as a secret leaf without the methods that redact it everywhere.
type leaf struct {
	value string
}

func (leaf) String() string   { return "leaf" }
func (leaf) GoString() string { return "leaf" }

// Accepted: listed as a redacting type that holds no secret.
type handle struct {
	digest [32]byte
}

func (h handle) String() string                 { return "handle([REDACTED])" }
func (h handle) GoString() string               { return h.String() }
func (h handle) Format(state fmt.State, _ rune) { _, _ = io.WriteString(state, h.String()) }
func (h handle) MarshalJSON() ([]byte, error)   { return json.Marshal(h.String()) }
func (h handle) LogValue() slog.Value           { return slog.StringValue(h.String()) }

// Accepted: holds only the non-secret handle.
type handleHolder struct {
	handle handle
}

// Accepted: listed as an exception.
type excepted struct {
	token contracts.SecretString
}

// Finding: another module's type that holds key material, through any field.
type tlsHolder struct {
	Config *tls.Config
}

// Accepted: the holder redacts itself.
type formattedHolder struct {
	token contracts.SecretString
}

func (formattedHolder) Format(state fmt.State, _ rune) { _, _ = io.WriteString(state, "[REDACTED]") }

// Accepted: a generic field is opaque until instantiated.
type genericBox[T any] struct {
	value T
}

// Finding: an instantiation that holds a secret in the box's unexported field.
type boxedSecret struct {
	box genericBox[contracts.SecretString]
}

// Accepted: the precise production field exception follows the actual recipe
// through an unexported holder; SecretFields contains validated variable names.
type loginRecipeMetadata struct {
	loginRecipes []pentestcontract.LoginRecipe
}

// Finding: an identically named field on a different holder contains secret
// values. The LoginRecipe exception must not exempt the field name generally.
type plainSecretFields struct {
	SecretFields []string
}
