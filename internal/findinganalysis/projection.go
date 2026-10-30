package findinganalysis

import (
	"mime"
	"net/url"
	"strings"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/contentdigest"
	"github.com/grauwolf32/contractor/internal/strictjson"
)

// Validate checks the closed resolved projection when reading an immutable
// snapshot. Exact source bytes were checked by Resolve before publication;
// this check never reopens or guesses a source.
func (r Resolved) Validate() error {
	if validate(r.Analysis) != nil || r.Sources == nil || len(r.Sources) != len(r.Analysis.Sources) || r.HTTP == nil || len(r.HTTP) != len(r.Analysis.HTTPExchanges) {
		return errInvalid
	}
	for i, s := range r.Sources {
		native := r.Analysis.Sources[i]
		id, err := auditdomain.FindingCollectionDocumentID(s.Document)
		if err != nil || id != s.Document.ID || s.ID != native.ID || s.Kind != native.Kind || !s.Document.Ref.SameExact(native.ArtifactRef) {
			return errInvalid
		}
		identity, err := strictjson.Canonical([]any{s.Document.Scope.Kind, s.Document.Scope.ID, s.Document.Ref.Namespace, s.Document.Ref.Name, *s.Document.Ref.Revision, s.Kind, s.Document.Digest})
		if err != nil || s.RootKey != strings.TrimPrefix(contentdigest.Bytes(identity), "sha256:") {
			return errInvalid
		}
	}
	for i, h := range r.HTTP {
		if h.ID != r.Analysis.HTTPExchanges[i].ID || h.Validate() != nil {
			return errInvalid
		}
	}
	return nil
}

// Validate ensures an HTTP export projection contains only its safe fields.
func (h ResolvedHTTP) Validate() error {
	if !identifier.MatchString(h.ID) || h.Origin != "producer-asserted" && h.Origin != "runtime-capture" || h.Status != nil && (*h.Status < 100 || *h.Status > 599) || len(h.Method) > 64 {
		return errInvalid
	}
	if h.URI != "" {
		u, err := url.Parse(h.URI)
		if err != nil || u.Scheme != "http" && u.Scheme != "https" || u.Host == "" || u.User != nil || u.RawQuery != "" || u.ForceQuery || u.Fragment != "" || u.Opaque != "" {
			return errInvalid
		}
	}
	if h.ContentType != "" {
		value, params, err := mime.ParseMediaType(h.ContentType)
		if err != nil || value != h.ContentType || len(params) != 0 {
			return errInvalid
		}
	}
	if len(h.Omitted) != 5 {
		return errInvalid
	}
	for i, expected := range []string{"bodies", "headers", "query", "fragment", "userinfo"} {
		if h.Omitted[i] != expected {
			return errInvalid
		}
	}
	return nil
}
