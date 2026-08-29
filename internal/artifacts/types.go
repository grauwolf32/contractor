package artifacts

import (
	"context"
	"regexp"
	"strings"

	"github.com/grauwolf32/contractor/internal/contracts"
)

const MaxPayloadSize = 16 * 1024 * 1024

type ScopeKind string

const (
	ScopeUser ScopeKind = "user"
	ScopeRun  ScopeKind = "run"
)

// Scope is created only through UserScope or RunScope; ArtifactRef never
// carries these fields.
type Scope struct {
	kind ScopeKind
	id   string
}

func UserScope(userID string) (Scope, error) { return newScope(ScopeUser, userID) }

func RunScope(runID string) (Scope, error) { return newScope(ScopeRun, runID) }

func newScope(kind ScopeKind, id string) (Scope, error) {
	if kind != ScopeUser && kind != ScopeRun || strings.TrimSpace(id) == "" || strings.ContainsRune(id, 0) {
		return Scope{}, ErrInvalidScope
	}
	return Scope{kind: kind, id: id}, nil
}

func (s Scope) Kind() ScopeKind { return s.kind }

func (s Scope) ID() string { return s.id }

type ArtifactRef = contracts.ArtifactRef

type Payload struct {
	MediaType string
	Data      []byte
}

type ReadResult struct {
	Ref     ArtifactRef
	Payload Payload
}

type WriteResult struct {
	Ref       ArtifactRef
	MediaType string
	Size      int64
}

type ForkResult struct {
	SourceRef ArtifactRef
	TargetRef ArtifactRef
	MediaType string
	Size      int64
}

type PinKind string

const (
	PinRunInput     PinKind = "run_input"
	PinStageContext PinKind = "stage_context"
	PinStageResult  PinKind = "stage_result"
	PinRunOutput    PinKind = "run_output"
)

// Repository is implemented by PostgreSQL and can be bound either to a pool
// or to a caller-owned transaction.
type Repository interface {
	Write(context.Context, Scope, ArtifactRef, Payload, *string) (WriteResult, error)
	Read(context.Context, Scope, ArtifactRef) (ReadResult, error)
	List(context.Context, Scope, *string) ([]ArtifactRef, error)
	ForkInput(context.Context, Scope, ArtifactRef, Scope, string) (ForkResult, error)
	BindOutputExact(context.Context, Scope, string, ArtifactRef, *string) (ForkResult, error)
	PinExact(context.Context, Scope, ArtifactRef, PinKind, string) error
	FreezeOutputs(context.Context, Scope) error
}

var mediaTypePattern = regexp.MustCompile("^[a-z0-9][a-z0-9!#$%&'+.^_`|~-]*/[a-z0-9][a-z0-9!#$%&'+.^_`|~-]*$")

func validateScope(scope Scope) error {
	if scope.kind != ScopeUser && scope.kind != ScopeRun || strings.TrimSpace(scope.id) == "" || strings.ContainsRune(scope.id, 0) {
		return ErrInvalidScope
	}
	return nil
}

func validateComponent(value string) error {
	if strings.TrimSpace(value) == "" || strings.Contains(value, "/") || strings.ContainsRune(value, 0) {
		return ErrInvalidName
	}
	return nil
}

func validateRef(ref ArtifactRef) error {
	if err := validateComponent(ref.Namespace); err != nil {
		return err
	}
	if err := validateComponent(ref.Name); err != nil {
		return err
	}
	if ref.Revision != nil {
		if err := validateRevision(*ref.Revision); err != nil {
			return err
		}
	}
	return nil
}

func validateRevision(value string) error {
	if strings.TrimSpace(value) == "" || strings.ContainsRune(value, 0) {
		return ErrInvalidName
	}
	return nil
}

func validateMediaType(value string) error {
	if value == "*/*" || !mediaTypePattern.MatchString(value) {
		return ErrInvalidMediaType
	}
	return nil
}

func validatePayload(payload Payload) error {
	if err := validateMediaType(payload.MediaType); err != nil {
		return err
	}
	if len(payload.Data) > MaxPayloadSize {
		return ErrPayloadTooLarge
	}
	return nil
}

func exactRevision(ref ArtifactRef) (string, error) {
	if err := validateRef(ref); err != nil {
		return "", err
	}
	if ref.Revision == nil {
		return "", ErrExactRevisionRequired
	}
	return *ref.Revision, nil
}
