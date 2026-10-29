// Package projectsitemap owns safe Project surface metadata and its deterministic
// reducer. ArtifactStore owns publication, history admission and exact retention.
package projectsitemap

import (
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/strictjson"
)

const (
	OperationTimeout         = 15 * time.Second
	Version                  = "project-sitemap@1"
	Projection               = "sitemap-safe-projection@1"
	MediaType                = "application/vnd.contractor.project-sitemap+json"
	OwnerSourceID            = "owner"
	MaxDocumentBytes         = 32 << 20
	MaxGroups                = 10000
	MaxMembers               = 50000
	MaxGroupMembers          = 64
	MaxMemberKeyBytes        = 8 << 10
	MaxExamples              = 8
	MaxExampleBytes          = 2 << 10
	MaxSummaries             = 16
	MaxProvenance            = 8
	MaxSources               = 128
	MaxRules                 = 256
	MaxMarkers               = 10000
	MaxNoteBytes             = 4 << 10
	MaxTags                  = 16
	MaxTagBytes              = 64
	MaxBatch                 = 100
	MaxBatchBytes            = 256 << 10
	MaxAttempts              = 3
	MaxOmissionCount  uint64 = 1<<53 - 1
)

// Document is the only persisted map representation. Arrays are canonical,
// sorted sets; omitted contributions never become invented concrete members.
type Document struct {
	Version    string   `json:"version" strictjson:"required"`
	Projection string   `json:"projection" strictjson:"required"`
	Groups     []Group  `json:"groups" strictjson:"required"`
	Sources    []Source `json:"sources" strictjson:"required"`
	Rules      []Rule   `json:"rules" strictjson:"required"`
	Markers    []Marker `json:"markers" strictjson:"required"`
	Gaps       []Gap    `json:"gaps" strictjson:"required"`
}

type Group struct {
	ID           string            `json:"id" strictjson:"required"`
	Origin       string            `json:"origin" strictjson:"required"`
	Method       string            `json:"method" strictjson:"required"`
	Route        string            `json:"route" strictjson:"required"`
	Structural   []StructuralValue `json:"structural" strictjson:"required"`
	Members      []Member          `json:"members" strictjson:"required"`
	Declarations []Declaration     `json:"declarations" strictjson:"required"`
	Omitted      uint64            `json:"omitted" strictjson:"required"`
}

// Member preserves the concrete operation independently of representative
// examples. Annotation and suppression selections follow ID through regrouping.
type Member struct {
	ID          string       `json:"id" strictjson:"required"`
	Path        string       `json:"path" strictjson:"required"`
	Summaries   []Summary    `json:"summaries" strictjson:"required"`
	Examples    []string     `json:"examples" strictjson:"required"`
	Evidence    []Evidence   `json:"evidence" strictjson:"required"`
	Annotations []Annotation `json:"annotations" strictjson:"required"`
	Omitted     uint64       `json:"omitted" strictjson:"required"`
}

// Provenance identifies the metadata source, not an execution or principal.
type Provenance struct {
	SourceID   string `json:"sourceId" strictjson:"required"`
	Kind       string `json:"kind" strictjson:"required"`
	Deployment string `json:"deployment,omitempty"`
}

// Summary aggregates observed behavior without attributing it to a user or session.
type Summary struct {
	Provenance Provenance `json:"provenance" strictjson:"required"`
	Outcome    string     `json:"outcome" strictjson:"required"`
	Status     int        `json:"status" strictjson:"required"`
	Parameters []string   `json:"parameters" strictjson:"required"`
	First      string     `json:"first" strictjson:"required"`
	Last       string     `json:"last" strictjson:"required"`
}

type Evidence struct {
	RunID    string                  `json:"runId" strictjson:"required"`
	Artifact contracts.ExactArtifact `json:"artifact" strictjson:"required"`
}

type Annotation struct {
	Provenance Provenance `json:"provenance" strictjson:"required"`
	Note       string     `json:"note" strictjson:"required"`
	Tags       []string   `json:"tags" strictjson:"required"`
}

type Declaration struct {
	Rule       bool       `json:"rule" strictjson:"required"`
	Provenance Provenance `json:"provenance" strictjson:"required"`
	Pointer    string     `json:"pointer" strictjson:"required"`
	Parameters []string   `json:"parameters" strictjson:"required"`
}

type Source struct {
	Provenance   Provenance               `json:"provenance" strictjson:"required"`
	Generation   uint64                   `json:"generation" strictjson:"required"`
	Artifact     *contracts.ExactArtifact `json:"artifact,omitempty"`
	PolicyDigest string                   `json:"policyDigest" strictjson:"required"`
}

type Rule struct {
	SourceID   string                `json:"sourceId" strictjson:"required"`
	Origin     string                `json:"origin" strictjson:"required"`
	Method     string                `json:"method" strictjson:"required"`
	Path       string                `json:"path" strictjson:"required"`
	Pointer    string                `json:"pointer" strictjson:"required"`
	Parameters []string              `json:"parameters" strictjson:"required"`
	Structural []StructuralParameter `json:"structural" strictjson:"required"`
}

// StructuralParameter is an explicit finite query selector. Neither a name nor
// a model suggestion alone authorizes retaining an arbitrary incoming value.
type StructuralParameter struct {
	Name    string   `json:"name" strictjson:"required"`
	Allowed []string `json:"allowed" strictjson:"required"`
}

type StructuralValue struct {
	Name   string   `json:"name" strictjson:"required"`
	Values []string `json:"values" strictjson:"required"`
}

// Marker records a fixed member selection. It never means every future member
// of a group that happens to acquire the same route.
type Marker struct {
	ID         string     `json:"id" strictjson:"required"`
	Members    []string   `json:"members" strictjson:"required"`
	Provenance Provenance `json:"provenance" strictjson:"required"`
}

// Gap is a bounded aggregate with no removed values, hashes or event IDs.
type Gap struct {
	SourceID string `json:"sourceId" strictjson:"required"`
	Reason   string `json:"reason" strictjson:"required"`
	Count    uint64 `json:"count" strictjson:"required"`
}

// Mutation is a typed command, never a replacement document. ExpectedRevision
// is mandatory for edits; observation batches rebase the same intent on CAS.
type Mutation struct {
	Operation        string                 `json:"operation" strictjson:"required"`
	ExpectedRevision string                 `json:"expectedRevision,omitempty"`
	Observations     []Observation          `json:"observations,omitempty"`
	Gaps             []Gap                  `json:"gaps,omitempty"`
	Artifact         *contracts.ArtifactRef `json:"artifact,omitempty"`
	URL              string                 `json:"url,omitempty"`
	Method           string                 `json:"method,omitempty"`
	MemberIDs        []string               `json:"memberIds,omitempty"`
	Note             string                 `json:"note,omitempty"`
	Tags             []string               `json:"tags,omitempty"`
	MarkerID         string                 `json:"markerId,omitempty"`
	Source           *SourceUpdate          `json:"source,omitempty"`
}

// UnmarshalJSON closes optional values too: omission is allowed, typed null is
// not. The alias makes strictjson check every nested member without recursion.
func (m *Mutation) UnmarshalJSON(data []byte) error {
	type wire Mutation
	var value wire
	if err := strictjson.DecodeNonNull(data, &value); err != nil {
		return ErrInvalid
	}
	*m = Mutation(value)
	return nil
}

type SourceUpdate struct {
	SourceID string                   `json:"sourceId" strictjson:"required"`
	Expected *Source                  `json:"expected,omitempty"`
	Artifact *contracts.ExactArtifact `json:"artifact,omitempty"`
	Rules    []Rule                   `json:"rules" strictjson:"required"`
}

// Observation is unprojected ingress. It deliberately accepts no headers or
// body values. Projection runs before any persisted ID or digest is computed.
type Observation struct {
	URL        string   `json:"url" strictjson:"required"`
	Method     string   `json:"method" strictjson:"required"`
	Outcome    string   `json:"outcome" strictjson:"required"`
	Status     int      `json:"status" strictjson:"required"`
	At         string   `json:"at" strictjson:"required"`
	Parameters []string `json:"parameters,omitempty"`
}

// Authority comes from the authenticated owner/allocation, never mutation JSON.
type Authority struct {
	OwnerID    string
	ProjectID  string
	RunID      string
	SourceID   string
	Kind       string
	Deployment string
	CanObserve bool
}

func Empty() Document {
	return Document{Version: Version, Projection: Projection, Groups: []Group{},
		Sources: []Source{}, Rules: []Rule{}, Markers: []Marker{}, Gaps: []Gap{}}
}
