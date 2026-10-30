package sarif

import (
	"bytes"
	"net/url"
	"regexp"
	"strings"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/findinganalysis"
	"github.com/grauwolf32/contractor/internal/findingexport"
	"github.com/grauwolf32/contractor/internal/strictjson"
)

const unclassifiedRule = "contractor/unclassified"

type registry struct {
	budget    *byteBudget
	rules     map[string]object
	ruleBytes map[string][]byte
	ruleIndex map[string]int
	roots     map[string]object
	files     map[string]object
	fileIndex map[string]int
	git       map[string]findingexport.GitSource
	taxa      map[string]map[string]object
	taxIndex  map[string]int
}

func newRegistry(budget *byteBudget) *registry {
	return &registry{budget: budget, rules: map[string]object{}, ruleBytes: map[string][]byte{}, ruleIndex: map[string]int{}, roots: map[string]object{}, files: map[string]object{}, fileIndex: map[string]int{}, git: map[string]findingexport.GitSource{}, taxa: map[string]map[string]object{}, taxIndex: map[string]int{}}
}

func ruleID(receipt findingexport.Receipt) string {
	if receipt.Analysis != nil && receipt.Analysis.Analysis.Rule != nil {
		return receipt.Analysis.Analysis.Rule.ID
	}
	return unclassifiedRule
}

func (r *registry) register(receipt findingexport.Receipt) error {
	id := ruleID(receipt)
	definition := object{"id": id, "shortDescription": object{"text": "Possible issue without a versioned check definition"}}
	if receipt.Analysis != nil && receipt.Analysis.Analysis.Rule != nil {
		rule := receipt.Analysis.Analysis.Rule
		definition["shortDescription"] = object{"text": rule.Name}
		definition["fullDescription"] = object{"text": rule.Description}
		definition["properties"] = object{"contractor": object{"rule_version": rule.Version, "producer_rule": rule}}
		if rule.HelpURI != "" {
			definition["helpUri"] = rule.HelpURI
		}
	}
	encoded, err := strictjson.Canonical(definition)
	if err != nil {
		return findingexport.ErrInvalid
	}
	if previous, exists := r.ruleBytes[id]; exists {
		if !bytes.Equal(previous, encoded) {
			return findingexport.ErrInvalid
		}
	} else {
		if err := r.budget.add(definition); err != nil {
			return err
		}
		r.rules[id], r.ruleBytes[id] = definition, encoded
	}
	for _, ref := range receipt.Proposal.StandardRefs {
		if !mappedTaxonomy(ref) {
			continue
		}
		key := taxonomyKey(ref)
		if r.taxa[key] == nil {
			r.taxa[key] = map[string]object{}
		}
		if _, exists := r.taxa[key][ref.RequirementID]; !exists {
			entry := object{"id": ref.RequirementID, "shortDescription": object{"text": ref.RequirementID}}
			if err := r.budget.add(entry); err != nil {
				return err
			}
			r.taxa[key][ref.RequirementID] = entry
		}
	}
	if receipt.Analysis == nil {
		return nil
	}
	evidence := map[string]findingexport.Evidence{}
	for _, item := range receipt.Evidence {
		evidence[item.Original.ID] = item
	}
	sources := map[string]findinganalysis.ResolvedSource{}
	for _, source := range receipt.Analysis.Sources {
		if _, exists := evidence[source.Document.ID]; !exists {
			return findingexport.ErrInvalid
		}
		sources[source.ID] = source
		base := baseID(source.RootKey)
		if _, exists := r.roots[base]; !exists {
			root := object{"uri": "contractor-source://" + source.RootKey + "/"}
			if err := r.budget.add(base); err != nil {
				return err
			}
			if err := r.budget.add(root); err != nil {
				return err
			}
			r.roots[base] = root
		}
		if origin := evidence[source.Document.ID].Git; origin != nil && source.Kind == "archive" && source.Document.MediaType == "application/zip" {
			if previous, exists := r.git[source.RootKey]; exists && previous != *origin {
				return findingexport.ErrInvalid
			}
			r.git[source.RootKey] = *origin
		}
	}
	for _, location := range receipt.Analysis.Analysis.Locations {
		source, exists := sources[location.SourceID]
		if !exists {
			return findingexport.ErrInvalid
		}
		key := fileKey(source.RootKey, location.Path)
		artifact := object{"location": object{"uri": relativeURI(location.Path), "uriBaseId": baseID(source.RootKey)}, "hashes": object{"sha-256": strings.TrimPrefix(location.FileDigest, "sha256:")}, "properties": object{"contractor": object{"source_kind": source.Kind, "source_document": source.Document, "source_root": source.RootKey}}}
		if source.Kind == "file" {
			artifact["mimeType"] = source.Document.MediaType
		}
		if previous, exists := r.files[key]; exists {
			a, errA := strictjson.Canonical(previous)
			b, errB := strictjson.Canonical(artifact)
			if errA != nil || errB != nil || !bytes.Equal(a, b) {
				return findingexport.ErrInvalid
			}
			continue
		}
		if err := r.budget.add(artifact); err != nil {
			return err
		}
		r.files[key] = artifact
	}
	return nil
}

func baseID(root string) string        { return "SRC_" + root }
func fileKey(root, path string) string { return root + "/" + path }
func relativeURI(path string) string {
	parts := strings.Split(path, "/")
	for i, part := range parts {
		parts[i] = url.PathEscape(part)
	}
	return strings.Join(parts, "/")
}

var cweID = regexp.MustCompile(`^CWE-[1-9][0-9]*$`)

// These adapters project already admitted frozen classifications; they do not
// replace the owning catalog/pinned-standard validation or infer aliases.
func mappedTaxonomy(ref auditdomain.StandardReference) bool {
	switch ref.Scheme {
	case "CWE":
		return ref.Version == "4.20" && cweID.MatchString(ref.RequirementID)
	case "asvs":
		return ref.Version == "4.0.3"
	case "owasp-asvs":
		return ref.Version == "5.0.0"
	default:
		return false
	}
}
func taxonomyKey(ref auditdomain.StandardReference) string { return ref.Scheme + "@" + ref.Version }

func (r *registry) metadata() (object, error) {
	rules := make([]any, 0, len(r.rules))
	for index, key := range sortedKeys(r.rules) {
		r.ruleIndex[key] = index
		rules = append(rules, r.rules[key])
	}
	files := make([]any, 0, len(r.files))
	for index, key := range sortedKeys(r.files) {
		r.fileIndex[key] = index
		files = append(files, r.files[key])
	}
	taxonomies := make([]any, 0, len(r.taxa))
	for index, key := range sortedKeys(r.taxa) {
		r.taxIndex[key] = index
		entries := make([]any, 0, len(r.taxa[key]))
		for _, id := range sortedKeys(r.taxa[key]) {
			entries = append(entries, r.taxa[key][id])
		}
		name, version, _ := strings.Cut(key, "@")
		component := object{"name": key, "version": version, "isComprehensive": false, "properties": object{"contractor": object{"scheme": name}}}
		if err := r.budget.add(component); err != nil {
			return nil, err
		}
		component["taxa"] = entries
		taxonomies = append(taxonomies, component)
	}
	vcs := make([]any, 0, len(r.git))
	for _, root := range sortedKeys(r.git) {
		source := r.git[root]
		entry := object{"repositoryUri": source.RepositoryURI, "revisionId": source.Commit, "mappedTo": object{"uri": "contractor-source://" + root + "/"}, "properties": object{"contractor": object{"source_root": root}}}
		if err := r.budget.add(entry); err != nil {
			return nil, err
		}
		vcs = append(vcs, entry)
	}
	driver := object{"name": "Contractor", "version": findingexport.ExporterVersion}
	// Classifications belong to results. SARIF 3.19.26 couples an optional
	// supportedTaxonomies claim to rule relationships, which native rules do
	// not declare. Do not infer those relationships from result tuples.
	if err := r.budget.add(driver); err != nil {
		return nil, err
	}
	driver["rules"] = rules
	run := object{"tool": object{"driver": driver}, "columnKind": "unicodeCodePoints"}
	if len(files) > 0 {
		run["artifacts"] = files
	}
	if len(r.roots) > 0 {
		run["originalUriBaseIds"] = r.roots
	}
	if len(vcs) > 0 {
		run["versionControlProvenance"] = vcs
	}
	if len(taxonomies) > 0 {
		run["taxonomies"] = taxonomies
	}
	return run, nil
}
