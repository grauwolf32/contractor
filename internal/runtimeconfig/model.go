// Package runtimeconfig owns immutable RuntimeConfig documents and the
// revisioned labels that select them. Values in this package are deliberately
// non-secret: credential fields contain identifiers only.
package runtimeconfig

import (
	"errors"
	"fmt"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
)

const (
	APIVersion = "contractor/v1alpha1"
	Kind       = "RuntimeConfig"

	BuiltInName    = "contractor-empty"
	BuiltInVersion = "1"
	BuiltInDigest  = "sha256:80a1754c01f8443c29fdc8f650a2254b2461694819918b204a55a7ad3425dc5f"
	DefaultLabel   = "default"

	BuiltInCanonicalDocument = `{"apiVersion":"contractor/v1alpha1","kind":"RuntimeConfig","metadata":{"name":"contractor-empty","version":"1"},"spec":{}}`
)

var (
	ErrInvalid      = errors.New("invalid RuntimeConfig")
	ErrNotFound     = errors.New("RuntimeConfig resource not found")
	ErrConflict     = errors.New("RuntimeConfig conflict")
	ErrPrecondition = errors.New("RuntimeConfig revision precondition failed")
	ErrReserved     = errors.New("reserved RuntimeConfig resource")
)

type Ref struct {
	Name    string `json:"name"`
	Version string `json:"version"`
	Digest  string `json:"digest"`
}

func (r Ref) String() string { return r.Name + "@" + r.Version + "#" + r.Digest }

type Field[T any] struct {
	Present bool
	Clear   bool
	Value   T
}

type AtomicPatch[T any] struct {
	Present bool
	Clear   bool
	Value   T
}

type LLMGatewayPatch struct {
	Present    bool
	Gateway    Field[contracts.LLMGatewayConfigRef]
	Credential Field[string]
}

type TelemetryConfig struct {
	Adapter             string `json:"adapter"`
	Endpoint            string `json:"endpoint"`
	Credential          string `json:"credential,omitempty"`
	CaptureContent      bool   `json:"captureContent"`
	FlushTimeoutSeconds int    `json:"flushTimeoutSeconds"`
}

type HTTPProxyConfig struct {
	Adapter     string   `json:"adapter"`
	ProxyURL    string   `json:"proxyUrl"`
	Credential  string   `json:"credential,omitempty"`
	CABundlePEM string   `json:"caBundlePem,omitempty"`
	Targets     []string `json:"targets"`
}

type WorkerPatch struct {
	LLMGateway LLMGatewayPatch
	Telemetry  AtomicPatch[TelemetryConfig]
	HTTPProxy  AtomicPatch[HTTPProxyConfig]
}

type PlannerPatch struct {
	Telemetry AtomicPatch[TelemetryConfig]
}

type Spec struct {
	Worker  WorkerPatch
	Planner PlannerPatch
}

func (s Spec) Empty() bool {
	return !s.Worker.LLMGateway.Present && !s.Worker.Telemetry.Present &&
		!s.Worker.HTTPProxy.Present && !s.Planner.Telemetry.Present
}

type Version struct {
	Ref               Ref
	Spec              Spec
	CanonicalDocument []byte
	BuiltIn           bool
	ActorID           string
	CreatedAt         time.Time
}

type Publication struct {
	IdempotencyKeyDigest string
	RequestDigest        string
	Ref                  Ref
	ActorID              string
	PublishedAt          time.Time
}

type Binding struct {
	Label     string
	Ref       Ref
	Revision  uint64
	CreatedBy string
	CreatedAt time.Time
	UpdatedBy string
	UpdatedAt time.Time
}

type LayerEntry struct {
	Label string
	Ref   Ref
	Spec  Spec
}

type MergeConflictError struct {
	Path string
	Refs []Ref
}

func (e *MergeConflictError) Error() string {
	return fmt.Sprintf("%v: same-layer conflict at %s between %d RuntimeConfigs", ErrConflict, e.Path, len(e.Refs))
}

func (e *MergeConflictError) Unwrap() error { return ErrConflict }
