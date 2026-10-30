package scanplan

import (
	"strings"

	"github.com/grauwolf32/contractor/internal/contentdigest"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/scan"
)

// sourceDocument is one exact OpenAPI source whose ref, bytes and document
// shape are valid: its parsed tree, its RequestSet identity (the exact ref and
// the digest of the supplied bytes), the media type that selected the parser
// and an index from each canonical path-item pointer to its key in paths.
//
// Lifetime and bounds (SCN-R256): one value serves one preparation call, either
// one operation or the operations of one AuditScanSettings.PrepareOperations
// batch, and nothing caches it across calls, settings or Runs. It holds one
// source within the document bounds (LIM-SCN-39, LIM-SCN-41, LIM-SCN-42) and
// an index with one entry per non-extension path key, which the node bound
// caps. It is never mutated after parseSource returns: preparers only read
// it, and options, reference and concrete-hint budgets, gaps and coverage
// belong to each operation's preparer.
type sourceDocument struct {
	identity  scan.RequestSetSource
	mediaType string
	root      map[string]any
	paths     map[string]any
	pathKeys  map[string]string
}

// The source parser and source digest are reached through these variables so
// that export_test.go can count source work; production never reassigns them.
var (
	parseSourceDocument = parseDocument
	sourceContentDigest = contentdigest.Bytes
)

// parseSource validates the exact ref and parses the whole source once. Its
// checks keep the order and codes of single-operation preparation: ref,
// media type and document bounds, OpenAPI version, then the paths mapping.
func parseSource(data []byte, mediaType string, ref contracts.ArtifactRef) (*sourceDocument, error) {
	if ref.ValidateExact() != nil {
		return nil, failure("invalid_source_ref")
	}
	root, err := parseSourceDocument(data, mediaType)
	if err != nil {
		return nil, err
	}
	version, _ := root["openapi"].(string)
	if !SupportsOpenAPIVersion(version) {
		return nil, failure("unsupported_openapi_version")
	}
	paths, ok := root["paths"].(map[string]any)
	if !ok {
		return nil, failure("invalid_document")
	}
	// JSON Pointer escaping is injective, so each pointer names at most one
	// path key. Extension members are never path items.
	pathKeys := make(map[string]string, len(paths))
	for path := range paths {
		if !strings.HasPrefix(path, "x-") {
			pathKeys["#/paths/"+escapePointer(path)] = path
		}
	}
	return &sourceDocument{
		identity:  scan.RequestSetSource{Artifact: ref.Clone(), ContentDigest: sourceContentDigest(data)},
		mediaType: mediaType, root: root, paths: paths, pathKeys: pathKeys,
	}, nil
}

// requestSetSource returns the source identity with its own revision, so no
// result aliases the caller's ref or another result of the same batch.
func (d *sourceDocument) requestSetSource() scan.RequestSetSource {
	return scan.RequestSetSource{Artifact: d.identity.Artifact.Clone(), ContentDigest: d.identity.ContentDigest}
}
