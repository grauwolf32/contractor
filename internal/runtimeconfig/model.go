// Package runtimeconfig owns immutable RuntimeConfig documents and the
// revisioned labels that select them. Values in this package are deliberately
// non-secret: credential fields contain identifiers only.
package runtimeconfig

import (
	"fmt"
	"time"

	"github.com/grauwolf32/contractor/internal/apierror"
	"github.com/grauwolf32/contractor/internal/contracts/llmgateway"
	"github.com/grauwolf32/contractor/internal/contracts/runtimesettings"
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

// ErrNotFound is the addressed resource that does not exist (404); a missing
// label or version that a request selects is the narrower ErrUnknownLabel or
// ErrVersionNotFound, which the public API answers as a request error.
var (
	ErrInvalid = apierror.New(apierror.RuntimeConfigInvalid, "invalid RuntimeConfig",
		apierror.WithMessage("Runtime label configuration is invalid"))
	ErrNotFound        = apierror.New(apierror.NotFound, "RuntimeConfig resource not found")
	ErrLabelNotFound   = fmt.Errorf("%w: Runtime label path binding", ErrNotFound)
	ErrUnknownLabel    = apierror.Narrow(ErrNotFound, apierror.RuntimeLabelUnknown, "selected Runtime label")
	ErrVersionNotFound = apierror.Narrow(ErrNotFound, apierror.RuntimeConfigInvalid, "selected RuntimeConfig version",
		apierror.WithMessage("selected RuntimeConfig version is unavailable"))
	ErrConflict = apierror.New(apierror.RuntimeConfigConflict, "RuntimeConfig conflict",
		apierror.WithMessage("selected Runtime labels conflict"))
	ErrPrecondition = apierror.New(apierror.PreconditionFailed, "RuntimeConfig revision precondition failed")
	ErrReserved     = apierror.New(apierror.RuntimeLabelInUse, "reserved RuntimeConfig resource",
		apierror.WithMessage("reserved Runtime label cannot be removed"))
	ErrPrincipalInUse = apierror.Narrow(ErrConflict, apierror.RuntimeAgentInUse, "Runtime Agent principal is in use")
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
	Gateway    Field[llmgateway.LLMGatewayConfigRef]
	Credential Field[string]
}

type TelemetryConfig struct {
	Adapter             string                                   `json:"adapter"`
	Endpoint            string                                   `json:"endpoint"`
	Credential          string                                   `json:"credential,omitempty"`
	CaptureContent      bool                                     `json:"captureContent"`
	FlushTimeoutSeconds int                                      `json:"flushTimeoutSeconds"`
	Export              *runtimesettings.TelemetryExportSettings `json:"export,omitempty"`
}

type HTTPProxyConfig struct {
	Adapter     string   `json:"adapter"`
	ProxyURL    string   `json:"proxyUrl"`
	Credential  string   `json:"credential,omitempty"`
	CABundlePEM string   `json:"caBundlePem,omitempty"`
	Targets     []string `json:"targets"`
}

type CaidoConfig struct {
	Adapter               string `json:"adapter"`
	Endpoint              string `json:"endpoint"`
	Credential            string `json:"credential,omitempty"`
	CABundlePEM           string `json:"caBundlePem,omitempty"`
	RequestTimeoutSeconds int    `json:"requestTimeoutSeconds,omitempty"`
}

type WorkerPatch struct {
	LLMGateway LLMGatewayPatch
	Telemetry  AtomicPatch[TelemetryConfig]
	HTTPProxy  AtomicPatch[HTTPProxyConfig]
	Caido      AtomicPatch[CaidoConfig]
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
		!s.Worker.HTTPProxy.Present && !s.Worker.Caido.Present && !s.Planner.Telemetry.Present
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

type LabelInUseError struct {
	RuntimeAgentIDs []string
}

func (e *LabelInUseError) Error() string {
	return "Runtime label is assigned to a Runtime Agent principal"
}
func (e *LabelInUseError) Unwrap() error { return ErrConflict }

// ErrorKind answers the error with runtime_label_in_use; its Runtime Agents
// are the details.
func (e *LabelInUseError) ErrorKind() apierror.Kind { return apierror.RuntimeLabelInUse }

// PublicMessage tells the caller why the label cannot be removed.
func (e *LabelInUseError) PublicMessage() string {
	return "Runtime label is assigned to a Runtime Agent"
}

func labelInUseError(principals []RuntimeAgentPrincipal) error {
	const maximumReferences = 128
	count := min(len(principals), maximumReferences)
	ids := make([]string, count)
	for index := 0; index < count; index++ {
		ids[index] = principals[index].RuntimeAgentID
	}
	return &LabelInUseError{RuntimeAgentIDs: ids}
}
