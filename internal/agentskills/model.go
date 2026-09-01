package agentskills

const (
	MediaType = "application/vnd.contractor.agent-skill+zip"

	MaximumArchiveBytes       = 16 << 20
	MaximumEntries            = 2000
	MaximumExpandedBytes      = 32 << 20
	MaximumManifestBytes      = 256 << 10
	MaximumFrontmatterBytes   = 32 << 10
	MaximumResourceBytes      = 1 << 20
	MaximumPathBytes          = 512
	MaximumPathComponents     = 8
	MaximumYAMLNodes          = 128
	MaximumYAMLDepth          = 3
	MaximumMetadataEntries    = 32
	MaximumDescriptionBytes   = 1024
	MaximumLicenseBytes       = 512
	MaximumCompatibilityBytes = 500
	MaximumMetadataValueBytes = 1024
)

// Manifest contains only bounded, safe metadata from SKILL.md.
type Manifest struct {
	Name          string            `json:"name"`
	Description   string            `json:"description"`
	License       string            `json:"license,omitempty"`
	Compatibility string            `json:"compatibility,omitempty"`
	Metadata      map[string]string `json:"metadata,omitempty"`
}

// Resource describes one validated non-manifest member.
type Resource struct {
	Path string `json:"path"`
	Size int64  `json:"size"`
}

// Member is a validated member with bounded bytes. Data returns a defensive
// copy so callers never need to reopen an untrusted raw ZIP member.
type Member struct {
	Path string
	data []byte
}

func (m Member) Size() int64 { return int64(len(m.data)) }

func (m Member) Data() []byte { return append([]byte(nil), m.data...) }

// Package is the all-or-nothing result of package validation.
type Package struct {
	Manifest      Manifest   `json:"manifest"`
	Digest        string     `json:"digest"`
	Resources     []Resource `json:"resources"`
	StoredBytes   int64      `json:"storedBytes"`
	ExpandedBytes int64      `json:"expandedBytes"`
	members       []Member
}

func (p *Package) Members() []Member {
	result := make([]Member, len(p.members))
	copy(result, p.members)
	return result
}

func (p *Package) Member(path string) (Member, bool) {
	for _, member := range p.members {
		if member.Path == path {
			return member, true
		}
	}
	return Member{}, false
}
