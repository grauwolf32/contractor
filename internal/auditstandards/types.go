// Package auditstandards owns immutable, versioned standard packages used by
// curated Audits. Package contents are trusted only after bounded validation;
// model output can reference, but never define, an authoritative entry.
package auditstandards

import (
	"encoding/json"

	"github.com/grauwolf32/contractor/internal/contracts"
)

const (
	Schema    = "contractor.audit-standard.v1"
	MediaType = "application/vnd.contractor.audit-standard+zip"

	CatalogNamespace = "audit-standards"
	ManifestPath     = "standard.json"

	MaximumPackages              = 64
	MaximumPackageBytes          = 8 << 20
	MaximumCatalogBytes          = 64 << 20
	MaximumManifestBytes         = 8 << 20
	MaximumEntries               = 10_000
	MaximumMappings              = 20_000
	MaximumEvidenceContracts     = 512
	MaximumEntryStatementBytes   = 16 << 10
	MaximumMappingObjectiveBytes = 16 << 10
)

type Reference struct {
	Scheme  string `json:"scheme"`
	Version string `json:"version"`
}

type DisclosurePolicy string

const (
	DisclosureMetadata    DisclosurePolicy = "metadata"
	DisclosureIdentifiers DisclosurePolicy = "identifiers"
	DisclosureFull        DisclosurePolicy = "full"
)

type Source struct {
	Name     string `json:"name"`
	URL      string `json:"url"`
	Revision string `json:"revision,omitempty"`
}

type License struct {
	ID          string           `json:"id"`
	URL         string           `json:"url"`
	Attribution string           `json:"attribution"`
	Disclosure  DisclosurePolicy `json:"disclosure"`
}

type Metadata struct {
	Scheme      string  `json:"scheme"`
	Version     string  `json:"version"`
	Title       string  `json:"title"`
	Description string  `json:"description"`
	Source      Source  `json:"source"`
	License     License `json:"license"`
}

func (m Metadata) Reference() Reference {
	return Reference{Scheme: m.Scheme, Version: m.Version}
}

type EvidenceContractRef struct {
	ID      string `json:"id"`
	Version string `json:"version"`
}

type EvidenceContract struct {
	ID                string   `json:"id"`
	Version           string   `json:"version"`
	Assessments       []string `json:"assessments"`
	EvidenceKinds     []string `json:"evidenceKinds"`
	MinimumEvidence   int      `json:"minimumEvidence"`
	MaximumEvidence   int      `json:"maximumEvidence"`
	HumanReview       string   `json:"humanReview"`
	RationaleRequired bool     `json:"rationaleRequired"`
}

func (c EvidenceContract) Reference() EvidenceContractRef {
	return EvidenceContractRef{ID: c.ID, Version: c.Version}
}

type Applicability struct {
	Mode string `json:"mode"`
	Rule string `json:"rule,omitempty"`
}

type Entry struct {
	ID               string              `json:"id"`
	Kind             string              `json:"kind"`
	Title            string              `json:"title"`
	Statement        string              `json:"statement"`
	Level            string              `json:"level,omitempty"`
	Applicability    Applicability       `json:"applicability"`
	AllowedMethods   []string            `json:"allowedMethods"`
	EvidenceContract EvidenceContractRef `json:"evidenceContract"`
}

type Mapping struct {
	Key              string              `json:"key"`
	EntryIDs         []string            `json:"entryIds"`
	WorkflowRole     string              `json:"workflowRole"`
	Method           string              `json:"method"`
	EvidenceContract EvidenceContractRef `json:"evidenceContract"`
	Title            string              `json:"title"`
	Objective        string              `json:"objective"`
}

type Document struct {
	Schema            string             `json:"schema"`
	Standard          Metadata           `json:"standard"`
	EvidenceContracts []EvidenceContract `json:"evidenceContracts"`
	Entries           []Entry            `json:"entries"`
	Mappings          []Mapping          `json:"mappings"`
}

type Package struct {
	Document      Document `json:"document"`
	Digest        string   `json:"digest"`
	StoredBytes   int64    `json:"storedBytes"`
	ExpandedBytes int64    `json:"expandedBytes"`
	payload       []byte
}

func (p Package) Reference() Reference { return p.Document.Standard.Reference() }

func (p Package) Payload() []byte { return append([]byte(nil), p.payload...) }

type ExactPackage struct {
	Artifact  contracts.ArtifactRef `json:"artifact"`
	Digest    string                `json:"digest"`
	MediaType string                `json:"mediaType"`
	SizeBytes int64                 `json:"sizeBytes"`
}

type ResolvedPackage struct {
	Package Package      `json:"package"`
	Source  ExactPackage `json:"source"`
}

// PinnedPackage is the complete historical identity placed in the Audit
// baseline. Retained points at the protected Audit-managed copy; Catalog is
// provenance and is not required to remain current or present afterwards.
type PinnedPackage struct {
	Reference Reference    `json:"reference"`
	Title     string       `json:"title"`
	Source    Source       `json:"source"`
	License   License      `json:"license"`
	Catalog   ExactPackage `json:"catalog"`
	Retained  ExactPackage `json:"retained"`
}

type EntryProjection struct {
	ID               string               `json:"id"`
	Kind             string               `json:"kind"`
	Title            string               `json:"title,omitempty"`
	Statement        string               `json:"statement,omitempty"`
	Level            string               `json:"level,omitempty"`
	Applicability    *Applicability       `json:"applicability,omitempty"`
	AllowedMethods   []string             `json:"allowedMethods,omitempty"`
	EvidenceContract *EvidenceContractRef `json:"evidenceContract,omitempty"`
}

type PackageProjection struct {
	Reference             Reference             `json:"reference"`
	Title                 string                `json:"title"`
	Description           string                `json:"description"`
	Source                Source                `json:"source"`
	License               License               `json:"license"`
	Digest                string                `json:"digest"`
	Artifact              contracts.ArtifactRef `json:"artifact"`
	EntryCount            int                   `json:"entryCount"`
	MappingCount          int                   `json:"mappingCount"`
	EvidenceContractCount int                   `json:"evidenceContractCount"`
	Entries               []EntryProjection     `json:"entries,omitempty"`
	EvidenceContracts     []EvidenceContract    `json:"evidenceContracts,omitempty"`
	Mappings              []Mapping             `json:"mappings,omitempty"`
}

type SeedStatus string

const (
	SeedCreated SeedStatus = "created"
	SeedInSync  SeedStatus = "in_sync"
)

type SeedMetadata struct {
	Reference Reference `json:"reference"`
	Digest    string    `json:"digest"`
	Size      int64     `json:"size"`
}

type SeedOutcome struct {
	Reference Reference  `json:"reference"`
	Status    SeedStatus `json:"status"`
	Digest    string     `json:"digest"`
}

type retainedProvenance struct {
	Schema  string       `json:"schema"`
	Catalog ExactPackage `json:"catalog"`
	Source  Source       `json:"source"`
	License License      `json:"license"`
}

func RetainedProvenance(pinned PinnedPackage) (json.RawMessage, error) {
	return json.Marshal(retainedProvenance{
		Schema:  "contractor.audit-standard-provenance.v1",
		Catalog: pinned.Catalog,
		Source:  pinned.Source,
		License: pinned.License,
	})
}
