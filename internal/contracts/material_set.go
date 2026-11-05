package contracts

import (
	"sort"
	"strings"
	"unicode/utf8"

	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
)

const (
	MaterialSetSchema          = "contractor.material-set.v1"
	MaxSelectedMaterials       = 32
	MaxMaterialDescriptorBytes = 1 << 20
)

// MaterialDescriptor is an exact scoped selection. Authority is checked by the
// caller; descriptors and their text never grant access to an Artifact.
type MaterialDescriptor struct {
	ScopeKind string      `json:"scope_kind" strictjson:"required"`
	ScopeID   string      `json:"scope_id" strictjson:"required"`
	Ref       ArtifactRef `json:"ref" strictjson:"required"`
	Digest    string      `json:"digest" strictjson:"required"`
	MediaType string      `json:"media_type" strictjson:"required"`
	SizeBytes int64       `json:"size_bytes" strictjson:"required"`
}

func (m MaterialDescriptor) Exact() ExactArtifact {
	return ExactArtifact{Ref: m.Ref, Digest: m.Digest, MediaType: m.MediaType, SizeBytes: m.SizeBytes}
}

func (m MaterialDescriptor) Validate() error {
	if m.ScopeKind != "user" && m.ScopeKind != "project" && m.ScopeKind != "audit" && m.ScopeKind != "run" {
		return Invalidf("invalid material scope")
	}
	if !utf8.ValidString(m.ScopeID) || strings.ContainsRune(m.ScopeID, 0) {
		return Invalidf("invalid material scope identifier")
	}
	if err := ValidateOpaqueID("material scope", m.ScopeID); err != nil {
		return err
	}
	if m.MediaType == "" {
		return Invalidf("material metadata is required")
	}
	return m.Exact().Validate()
}

// MaterialKey orders and deduplicates complete exact selections, preserving
// independent sources and different revisions even when bytes happen to match.
func MaterialKey(m MaterialDescriptor) string {
	b, _ := strictjson.Canonical(m)
	return string(b)
}

func MaterialHandle(m MaterialDescriptor) string {
	return "material-" + contentdigest.Bytes([]byte(MaterialKey(m)))[7:]
}

type MaterialSet struct {
	Schema    string               `json:"schema" strictjson:"required"`
	Materials []MaterialDescriptor `json:"materials" strictjson:"required"`
}

// DecodeMaterialSet checks the common bounded selection without resolving refs.
// Input order and repeated exact descriptors do not change the normalized set.
func DecodeMaterialSet(data []byte) (MaterialSet, error) {
	if len(data) == 0 || len(data) > MaxMaterialDescriptorBytes {
		return MaterialSet{}, Invalidf("material descriptor bytes exceeded")
	}
	var set MaterialSet
	if err := strictjson.Decode(data, &set); err != nil {
		return MaterialSet{}, Invalidf("invalid material set syntax or members")
	}
	if set.Schema != MaterialSetSchema || set.Materials == nil {
		return MaterialSet{}, Invalidf("invalid material set")
	}
	budget := MaterialBudget{}
	for _, m := range set.Materials {
		if err := budget.Add(m); err != nil {
			return MaterialSet{}, err
		}
	}
	set.Materials = budget.Materials()
	return set, nil
}

// MaterialBudget applies one cumulative count/byte bound across sets and
// singleton research materials. System context is accounted by its own owner.
type MaterialBudget struct {
	selected      map[string]MaterialDescriptor
	bytes         int
	documentBytes int64
	documents     map[string]bool
}

// AddSetDocument bounds total selection-document transfer before any payload
// read. Rechecking the same exact document within preparation is free.
func (b *MaterialBudget) AddSetDocument(m MaterialDescriptor, bytes int64) error {
	key := MaterialKey(m)
	if b.documents == nil {
		b.documents = map[string]bool{}
	}
	if b.documents[key] {
		return nil
	}
	if bytes < 0 || b.documentBytes+bytes > MaxMaterialDescriptorBytes {
		return Invalidf("material descriptor document bytes exceeded")
	}
	b.documents[key] = true
	b.documentBytes += bytes
	return nil
}

func (b *MaterialBudget) Add(m MaterialDescriptor) error {
	if err := m.Validate(); err != nil {
		return err
	}
	key := MaterialKey(m)
	if b.selected == nil {
		b.selected = map[string]MaterialDescriptor{}
	}
	if _, ok := b.selected[key]; ok {
		return nil
	}
	if len(b.selected) >= MaxSelectedMaterials || b.bytes+len(key) > MaxMaterialDescriptorBytes {
		return Invalidf("selected material limit exceeded")
	}
	b.selected[key] = m
	b.bytes += len(key)
	return nil
}
func (b *MaterialBudget) Materials() []MaterialDescriptor {
	keys := make([]string, 0, len(b.selected))
	for k := range b.selected {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	result := make([]MaterialDescriptor, 0, len(keys))
	for _, k := range keys {
		result = append(result, b.selected[k])
	}
	return result
}
