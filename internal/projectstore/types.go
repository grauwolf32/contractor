package projectstore

import "time"

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

type Project struct {
	ProjectID   string
	OwnerID     string
	Kind        Kind
	Name        string
	Description string
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
}

type ListParams struct {
	OwnerID         string
	Kind            *Kind
	BeforeCreatedAt *time.Time
	BeforeProjectID string
	Limit           int
}
