package contracts

import (
	"encoding/json"
	"slices"

	"go.yaml.in/yaml/v4"
)

const (
	SitemapAccessVersion     = "sitemap-access@1"
	MaxSitemapSnapshotInputs = 8
	SitemapListTool          = "list_sitemap_entries"
	SitemapReadTool          = "read_sitemap_entry"
	SitemapUpsertTool        = "upsert_sitemap_entry"
	SitemapRemoveTool        = "remove_sitemap_entry"
	SitemapRestoreTool       = "restore_sitemap_entry"
	SitemapGroupingTool      = "configure_sitemap_grouping"
	SitemapSnapshotTool      = "snapshot_sitemap_context"
	SitemapImportTool        = "import_sitemap_source"
)

func SitemapToolNames() []string {
	return []string{SitemapListTool, SitemapReadTool, SitemapUpsertTool, SitemapRemoveTool, SitemapRestoreTool, SitemapGroupingTool, SitemapSnapshotTool, SitemapImportTool}
}

// ProjectSitemapPolicy is pinned configuration, not a model-supplied context.
// Selected operations decide reader/editor/importer access independently of
// the explicit permission for the allocation's automatic observation writer.
type ProjectSitemapPolicy struct {
	Version             string   `json:"version" yaml:"version" strictjson:"required"`
	CollectObservations bool     `json:"collectObservations" yaml:"collectObservations" strictjson:"required"`
	Deployment          *string  `json:"deployment,omitempty" yaml:"deployment,omitempty"`
	SnapshotInputs      []string `json:"snapshotInputs,omitempty" yaml:"snapshotInputs,omitempty"`
}

func (p ProjectSitemapPolicy) Validate() error {
	if p.Version != SitemapAccessVersion ||
		p.Deployment != nil && !ValidResourceID(*p.Deployment) || len(p.SnapshotInputs) > MaxSitemapSnapshotInputs {
		return Invalidf("invalid Project sitemap policy")
	}
	if !slices.IsSorted(p.SnapshotInputs) || len(slices.Compact(slices.Clone(p.SnapshotInputs))) != len(p.SnapshotInputs) {
		return Invalidf("sitemap snapshot inputs must be sorted and unique")
	}
	for _, name := range p.SnapshotInputs {
		if !ValidName(name) {
			return Invalidf("invalid sitemap snapshot input name")
		}
	}
	return nil
}

func (p *ProjectSitemapPolicy) Clone() *ProjectSitemapPolicy {
	if p == nil {
		return nil
	}
	copy := *p
	if p.Deployment != nil {
		deployment := *p.Deployment
		copy.Deployment = &deployment
	}
	copy.SnapshotInputs = slices.Clone(p.SnapshotInputs)
	return &copy
}

func (p *ProjectSitemapPolicy) UnmarshalJSON(data []byte) error {
	type wire ProjectSitemapPolicy
	var value wire
	if _, err := DecodeStrictObject(data, &value); err != nil {
		return err
	}
	*p = ProjectSitemapPolicy(value)
	return p.Validate()
}

func (p *ProjectSitemapPolicy) UnmarshalYAML(node *yaml.Node) error {
	var fields map[string]any
	if err := node.Load(&fields, yaml.WithUniqueKeys()); err != nil {
		return err
	}
	data, err := json.Marshal(fields)
	if err != nil {
		return Invalidf("sitemap policy is not JSON-compatible")
	}
	// Set order is normalized at authoring; private wire readers remain strict.
	if names, ok := fields["snapshotInputs"].([]any); ok {
		strings := make([]string, 0, len(names))
		for _, name := range names {
			text, valid := name.(string)
			if !valid {
				return Invalidf("invalid sitemap snapshot input name")
			}
			strings = append(strings, text)
		}
		slices.Sort(strings)
		fields["snapshotInputs"] = strings
		data, err = json.Marshal(fields)
		if err != nil {
			return err
		}
	}
	return p.UnmarshalJSON(data)
}
