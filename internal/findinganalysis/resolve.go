package findinganalysis

import (
	"archive/zip"
	"bytes"
	"context"
	"fmt"
	"io"
	"strings"
	"unicode/utf8"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/contentdigest"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/sourcezip"
	"github.com/grauwolf32/contractor/internal/strictjson"
	"github.com/grauwolf32/contractor/internal/zipdirectory"
)

// Resolved retains native producer assertions independently of physical source
// validation. A validated source never upgrades an assertion to a tool receipt.
type Resolved struct {
	Analysis Document         `json:"analysis"`
	Sources  []ResolvedSource `json:"sources"`
	HTTP     []ResolvedHTTP   `json:"http"`
}

type ResolvedSource struct {
	ID       string                                `json:"id"`
	Kind     string                                `json:"kind"`
	Document auditdomain.FindingCollectionDocument `json:"document"`
	RootKey  string                                `json:"root_key"`
}

type evidenceSet struct {
	docs     []auditdomain.FindingCollectionDocument
	contents map[string][]byte
	origins  map[string]auditdomain.FindingCollectionDocument
}

// Resolver owns one publication's bounded evidence caches. Repeated receipts
// sharing a source verify and expand its bytes once, rather than once per result.
// Discard the resolver after publication, failure or cancellation.
type Resolver struct {
	verified      map[string]bool
	archives      map[string]*archiveIndex
	expanded      uint64
	selectedBytes int64
}

type archiveIndex struct {
	members map[string]*zip.File
	files   map[string][]byte
}

// Resolve consumes only the containing receipt's explicitly selected documents.
// It never follows a ref into another receipt or performs another store read.
// Absence of analysis is valid and returns nil; invalid analysis fails whole.
func (r *Resolver) Resolve(ctx context.Context, clientKey string, documents []auditdomain.FindingCollectionDocument, contents map[string][]byte, origins map[string]auditdomain.FindingCollectionDocument) (*Resolved, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if r.verified == nil {
		r.verified = map[string]bool{}
		r.archives = map[string]*archiveIndex{}
	}
	evidence := evidenceSet{docs: documents, contents: contents, origins: origins}
	var analysisDoc *auditdomain.FindingCollectionDocument
	var total int64
	for i := range documents {
		doc := &documents[i]
		data, ok := contents[doc.ID]
		if !ok || doc.Ref.ValidateExact() != nil || int64(len(data)) != doc.SizeBytes {
			return nil, errInvalid
		}
		id, err := auditdomain.FindingCollectionDocumentID(*doc)
		if err != nil || id != doc.ID {
			return nil, errInvalid
		}
		if origin, present := origins[doc.ID]; present {
			id, err := auditdomain.FindingCollectionDocumentID(origin)
			if err != nil || id != origin.ID || origin.Digest != doc.Digest || origin.MediaType != doc.MediaType || origin.SizeBytes != doc.SizeBytes {
				return nil, errInvalid
			}
		}
		if !r.verified[doc.ID] {
			if contentdigest.Bytes(data) != doc.Digest {
				return nil, errInvalid
			}
			if len(r.verified) >= auditdomain.MaximumCollectionDocuments || doc.SizeBytes > auditdomain.MaximumArchiveBytes-r.selectedBytes {
				return nil, errInvalid
			}
			r.selectedBytes += doc.SizeBytes
			r.verified[doc.ID] = true
		}
		total += doc.SizeBytes
		if total > auditdomain.MaximumArchiveBytes {
			return nil, errInvalid
		}
		for j := 0; j < i; j++ {
			if doc.Ref.SameExact(documents[j].Ref) {
				return nil, errInvalid
			}
		}
		if doc.MediaType == MediaType {
			if analysisDoc != nil {
				return nil, errInvalid
			}
			analysisDoc = doc
		}
	}
	if analysisDoc == nil {
		return nil, nil
	}
	native, err := decode(contents[analysisDoc.ID])
	if err != nil || native.ClientKey != clientKey {
		return nil, errInvalid
	}
	result := &Resolved{Analysis: native, Sources: []ResolvedSource{}, HTTP: []ResolvedHTTP{}}
	// Sources are opened once. File bodies and ZIP members are retained only for
	// the duration of this resolver; the snapshot keeps hashes/coordinates only.
	for _, source := range native.Sources {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		doc, data, ok := evidence.get(source.ArtifactRef)
		if !ok || doc.ID == analysisDoc.ID {
			return nil, errInvalid
		}
		wanted := map[string]bool{}
		for _, location := range native.Locations {
			if location.SourceID == source.ID {
				wanted[location.Path] = true
			}
		}
		files := map[string][]byte{}
		if source.Kind == "archive" {
			if doc.MediaType != "application/zip" {
				return nil, errInvalid
			}
			index := r.archives[doc.ID]
			if index == nil {
				index, err = indexArchive(ctx, data, &r.expanded)
				if err != nil {
					return nil, err
				}
				r.archives[doc.ID] = index
			}
			files, err = index.read(ctx, wanted)
			if err != nil {
				return nil, err
			}
		} else {
			if len(data) > sourcezip.MaxFileBytes || !utf8.Valid(data) || len(wanted) > 1 {
				return nil, errInvalid
			}

			for path := range wanted {
				files[path] = data
			}
		}
		for _, location := range native.Locations {
			if location.SourceID != source.ID {
				continue
			}
			data, found := files[location.Path]
			if !found || !utf8.Valid(data) || contentdigest.Bytes(data) != location.FileDigest || !validRegion(data, location.Region) {
				return nil, errInvalid
			}
		}
		identity, err := strictjson.Canonical([]any{doc.Scope.Kind, doc.Scope.ID, doc.Ref.Namespace, doc.Ref.Name, *doc.Ref.Revision, source.Kind, doc.Digest})
		if err != nil {
			return nil, errInvalid
		}
		result.Sources = append(result.Sources, ResolvedSource{ID: source.ID, Kind: source.Kind, Document: doc, RootKey: strings.TrimPrefix(contentdigest.Bytes(identity), "sha256:")})
	}
	// Every dependency is explicit, bounded, exact and non-recursive. Analysis
	// cannot cite itself or another analysis to invent a new evidence authority.
	refs := func(values []contracts.ArtifactRef) bool {
		if len(values) > auditdomain.MaximumCollectionDocuments {
			return false
		}
		for i, ref := range values {
			doc, _, ok := evidence.get(ref)
			if !ok || doc.MediaType == MediaType {
				return false
			}
			for j := 0; j < i; j++ {
				if ref.SameExact(values[j]) {
					return false
				}
			}
		}
		return true
	}
	for _, flow := range native.Flows {
		if !refs(flow.EvidenceRefs) {
			return nil, errInvalid
		}
		for _, step := range flow.Steps {
			if !refs(step.EvidenceRefs) {
				return nil, errInvalid
			}
		}
		if flow.Kind == "data-flow" && flow.Completeness == "complete" {
			for _, step := range flow.Steps {
				located := false
				for _, location := range native.Locations {
					if location.ID == step.LocationID && location.Region != nil {
						located = true
						break
					}
				}
				if !located {
					return nil, errInvalid
				}
			}
		}
		for _, edge := range flow.Edges {
			if !refs(edge.EvidenceRefs) {
				return nil, errInvalid
			}
			// No current durable code-analysis decoder proves a call/data-flow edge.
			// Copied graph responses and annotation text remain producer assertions.
			if edge.Basis == "tool-reported" {
				return nil, errInvalid
			}
		}
	}
	for _, exchange := range native.HTTPExchanges {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		doc, data, ok := evidence.get(exchange.EvidenceRef)
		if !ok || doc.MediaType == MediaType {
			return nil, errInvalid
		}
		http, err := resolveHTTP(exchange, doc, data)
		if err != nil {
			return nil, err
		}
		if exchange.Operation != nil {
			if err := resolveOperation(*exchange.Operation, evidence); err != nil {
				return nil, err
			}
		}
		result.HTTP = append(result.HTTP, http)
	}
	return result, nil
}

func (s evidenceSet) get(ref contracts.ArtifactRef) (auditdomain.FindingCollectionDocument, []byte, bool) {
	if ref.ValidateExact() != nil {
		return auditdomain.FindingCollectionDocument{}, nil, false
	}
	for _, stored := range s.docs {
		origin := stored
		if original, present := s.origins[stored.ID]; present {
			origin = original
		}
		if origin.Ref.SameExact(ref) {
			return origin, s.contents[stored.ID], true
		}
	}
	return auditdomain.FindingCollectionDocument{}, nil, false
}

func indexArchive(ctx context.Context, data []byte, expanded *uint64) (*archiveIndex, error) {
	if _, err := zipdirectory.Check(data, sourcezip.MaxEntries); err != nil {
		return nil, errInvalid
	}
	reader, err := zip.NewReader(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		return nil, errInvalid
	}
	index := &archiveIndex{members: map[string]*zip.File{}, files: map[string][]byte{}}
	seen := map[string]bool{}
	for _, file := range reader.File {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		name := strings.TrimSuffix(file.Name, "/")
		if !validPath(name) || seen[name] || !file.Mode().IsRegular() && !file.FileInfo().IsDir() || file.UncompressedSize64 > sourcezip.MaxFileBytes {
			return nil, errInvalid
		}
		seen[name] = true
		*expanded += file.UncompressedSize64
		if *expanded > sourcezip.MaxArchiveBytes {
			return nil, errInvalid
		}
		if file.Mode().IsRegular() {
			index.members[file.Name] = file
		}
	}
	for name := range index.members {
		parts := strings.Split(name, "/")
		for end := 1; end < len(parts); end++ {
			if index.members[strings.Join(parts[:end], "/")] != nil {
				return nil, errInvalid
			}
		}
	}
	return index, nil
}

func (index *archiveIndex) read(ctx context.Context, wanted map[string]bool) (map[string][]byte, error) {
	files := make(map[string][]byte, len(wanted))
	for name := range wanted {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if data, ok := index.files[name]; ok {
			files[name] = data
			continue
		}
		member := index.members[name]
		if member == nil {
			return nil, errInvalid
		}
		stream, err := member.Open()
		if err != nil {
			return nil, errInvalid
		}
		data, readErr := io.ReadAll(io.LimitReader(stream, sourcezip.MaxFileBytes+1))
		closeErr := stream.Close()
		if readErr != nil || closeErr != nil || uint64(len(data)) != member.UncompressedSize64 || len(data) > sourcezip.MaxFileBytes {
			return nil, errInvalid
		}
		index.files[name] = data
		files[name] = data
	}
	return files, nil
}

func validRegion(data []byte, region *Region) bool {
	if region == nil {
		return true
	}
	if len(data) == 0 {
		return false
	}
	// CRLF is one terminator. A final terminator permits the following empty
	// line; tabs count as one Unicode code point, never display columns.
	text := strings.ReplaceAll(string(data), "\r\n", "\n")
	text = strings.ReplaceAll(text, "\r", "\n")
	lines := strings.Split(text, "\n")
	end := region.StartLine
	if region.EndLine != nil {
		end = *region.EndLine
	}
	if region.StartLine < 1 || end < region.StartLine || end > int64(len(lines)) {
		return false
	}
	if region.StartColumn != nil && *region.StartColumn > int64(utf8.RuneCountInString(lines[region.StartLine-1]))+1 {
		return false
	}
	if region.EndColumn != nil && *region.EndColumn > int64(utf8.RuneCountInString(lines[end-1]))+1 {
		return false
	}
	return true
}

// invalidEvidence is deliberately fixed text: decoder exceptions, paths, source
// text and stripped HTTP values never escape in diagnostics.
func invalidEvidence() error { return fmt.Errorf("%w: evidence", errInvalid) }
