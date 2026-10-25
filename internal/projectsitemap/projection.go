package projectsitemap

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/scan"
	"net/url"
	"regexp"
	"slices"
	"strings"
	"unicode"
	"unicode/utf8"
)

var (
	secretName    = regexp.MustCompile(`(?i)(password|passwd|secret|token|session|cookie|authorization|credential|api[-_]?key|access[-_]?key|csrf|xsrf|nonce|signature|jwt|bearer)`)
	opaqueValue   = regexp.MustCompile(`[A-Za-z0-9_+~=-]{24,}|[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+`)
	methodPattern = regexp.MustCompile(`^[A-Z]{1,16}$`)
	hashPattern   = regexp.MustCompile(`^[0-9a-f]{64}$`)
)

func key(parts ...string) string {
	h := sha256.New()
	for _, part := range parts {
		_, _ = fmt.Fprintf(h, "%d:%s", len(part), part)
	}
	return hex.EncodeToString(h.Sum(nil))
}

// safeText rejects rather than hashes secret-bearing values. Repeated URL
// decoding catches percent-encoded credentials before public identity exists.
func safeText(value string, secrets []contracts.SecretString) bool {
	if !utf8.ValidString(value) || strings.ContainsRune(value, 0) {
		return false
	}
	for range 4 {
		if opaqueValue.MatchString(value) {
			return false
		}
		for _, secret := range secrets {
			if plain := secret.Reveal(); plain != "" && strings.Contains(value, plain) {
				return false
			}
		}
		decoded, err := url.PathUnescape(value)
		if err != nil {
			return false
		}
		if decoded == value {
			return true
		}
		value = decoded
	}
	return false
}

func projectURL(raw string, secrets []contracts.SecretString) (origin, path string, parameters []string, err error) {
	if len(raw) > MaxMemberKeyBytes || !utf8.ValidString(raw) {
		return "", "", nil, ErrInvalid
	}
	u, parseErr := url.Parse(raw)
	if parseErr != nil || u.Host == "" || (u.Scheme != "http" && u.Scheme != "https") || u.Opaque != "" {
		return "", "", nil, ErrInvalid
	}
	// Userinfo, query values and fragments are never copied, including in IDs.
	u.User, u.Fragment, u.RawFragment = nil, "", ""
	request := scan.PreparedHTTPRequest{Method: "GET", URL: u.String(), Headers: []scan.HTTPRequestHeader{}}
	if request.Validate() != nil || !scan.ValidHost(u.Hostname()) {
		return "", "", nil, ErrInvalid
	}
	origin = strings.ToLower(u.Scheme + "://" + u.Host)
	if strings.HasSuffix(origin, ":80") && u.Scheme == "http" {
		origin = strings.TrimSuffix(origin, ":80")
	}
	if strings.HasSuffix(origin, ":443") && u.Scheme == "https" {
		origin = strings.TrimSuffix(origin, ":443")
	}
	path = u.EscapedPath()
	if path == "" {
		path = "/"
	}
	if !safeText(origin, secrets) || !safeText(path, secrets) {
		return "", "", nil, ErrUnsafe
	}
	segments := strings.Split(path, "/")
	for index, segment := range segments {
		decoded, _ := url.PathUnescape(segment)
		// A credential-bearing path cannot supply a truthful concrete member.
		if secretName.MatchString(decoded) && index+1 < len(segments) && segments[index+1] != "" {
			return "", "", nil, ErrUnsafe
		}
	}
	parameters = []string{}
	query, parseErr := url.ParseQuery(u.RawQuery)
	if parseErr != nil {
		return "", "", nil, ErrInvalid
	}
	if len(query) > MaxBatch {
		return "", "", nil, ErrInvalid
	}
	for name := range query {
		if safeName(name, secrets) {
			parameters = append(parameters, name)
		}
	}
	slices.Sort(parameters)
	return origin, path, parameters, nil
}

func projectedLocations(raw string) bool {
	u, err := url.Parse(raw)
	return err == nil && (u.User != nil || u.RawQuery != "" || u.Fragment != "")
}

func safeName(name string, secrets []contracts.SecretString) bool {
	return len(name) > 0 && len(name) <= MaxTagBytes && safeText(name, secrets) &&
		!secretName.MatchString(name) && !strings.ContainsFunc(name, unicode.IsControl)
}

func projectAnnotation(note string, tags []string, a Authority) (Annotation, bool, error) {
	if len(note) > MaxNoteBytes || len(tags) > MaxTags {
		return Annotation{}, false, ErrInvalid
	}
	annotation := Annotation{Provenance: a.provenance(), Tags: []string{}}
	omitted := false
	if safeText(note, a.Secrets) && !secretName.MatchString(note) {
		annotation.Note = note
	} else {
		omitted = true
	}
	for _, tag := range tags {
		if len(tag) > MaxTagBytes {
			return Annotation{}, false, ErrInvalid
		}
		if safeName(tag, a.Secrets) {
			annotation.Tags = append(annotation.Tags, tag)
		} else {
			omitted = true
		}
	}
	slices.Sort(annotation.Tags)
	annotation.Tags = slices.Compact(annotation.Tags)
	return annotation, omitted, nil
}

func (a Authority) provenance() Provenance {
	return Provenance{SourceID: a.SourceID, Kind: a.Kind, Mode: a.Mode, RunID: a.RunID, Deployment: a.Deployment, CredentialRevision: a.CredentialRevision, SessionGeneration: a.SessionGeneration}
}

func (a Authority) contextKey() string {
	if a.IdentityID == "" {
		return key("unknown", a.Engagement, a.RunID)
	}
	if a.Engagement != "" {
		return key("engagement", a.Engagement, a.IdentityID)
	}
	return key("run", a.RunID, a.IdentityID)
}

func (a Authority) validate() error {
	if !contracts.ValidResourceID(a.SourceID) || (a.Mode != "blackbox" && a.Mode != "source-assisted") ||
		!slices.Contains([]string{"manual", "http", "caido", "ffuf", "katana", "dirsearch", "openapi"}, a.Kind) {
		return ErrInvalid
	}
	for _, value := range []string{a.RunID, a.Deployment, a.Engagement, a.IdentityID, a.CredentialRevision, a.SessionGeneration} {
		if value != "" && !contracts.ValidResourceID(value) {
			return ErrInvalid
		}
	}
	if a.CanObserve && a.RunID == "" {
		return ErrInvalid
	}
	if a.CanObserve && !observedKind(a.Kind) {
		return ErrInvalid
	}
	return nil
}

func observedKind(kind string) bool {
	return slices.Contains([]string{"http", "caido", "ffuf", "katana", "dirsearch"}, kind)
}
