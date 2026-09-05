package projectstore

import (
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
)

const (
	MaxNameBytes        = 160
	MaxDescriptionBytes = 4096
	MaxPageSize         = 200
)

type Kind string

const (
	KindProject    Kind = "project"
	KindEvaluation Kind = "evaluation"
)

func (k Kind) Valid() bool { return k == KindProject || k == KindEvaluation }

type Lifecycle string

const (
	LifecycleActive   Lifecycle = "active"
	LifecycleDeleting Lifecycle = "deleting"
)

func (l Lifecycle) Valid() bool { return l == LifecycleActive || l == LifecycleDeleting }

type DeletionPhase string

const (
	DeletionCancelling       DeletionPhase = "cancelling"
	DeletionDraining         DeletionPhase = "draining"
	DeletionPurgingRuns      DeletionPhase = "purging_runs"
	DeletionPurgingArtifacts DeletionPhase = "purging_artifacts"
)

func (p DeletionPhase) Valid() bool {
	return p == DeletionCancelling || p == DeletionDraining ||
		p == DeletionPurgingRuns || p == DeletionPurgingArtifacts
}

type Deletion struct {
	Phase       DeletionPhase
	RequestedAt time.Time
}

type Project struct {
	ProjectID   string
	OwnerID     string
	Kind        Kind
	Name        string
	Description string
	HTTPTarget  *contracts.HTTPOriginTargetRef
	Lifecycle   Lifecycle
	Deletion    *Deletion
	Revision    uint64
	CreatedAt   time.Time
	UpdatedAt   time.Time
}

type CreateParams struct {
	ProjectID      string
	OwnerID        string
	Kind           Kind
	Name           string
	Description    string
	IdempotencyKey string
	RequestDigest  string
}

type UpdateParams struct {
	ProjectID        string
	OwnerID          string
	ExpectedRevision uint64
	Name             string
	Description      string
	HTTPTarget       *contracts.HTTPOriginTargetRef
}

type BeginDeletionParams struct {
	ProjectID        string
	OwnerID          string
	ExpectedRevision uint64
}

type ListParams struct {
	OwnerID         string
	Kind            *Kind
	BeforeCreatedAt *time.Time
	BeforeProjectID string
	Limit           int
}
