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

// validText checks representation only, without classifying metadata as secrets.
func validText(value string) bool {
	return utf8.ValidString(value) && !strings.ContainsRune(value, 0)
}

func projectURL(raw string) (origin, path string, parameters []string, err error) {
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
	if !validText(path) {
		return "", "", nil, ErrInvalid
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
		if validLabel(name) {
			parameters = append(parameters, name)
		}
	}
	slices.Sort(parameters)
	return origin, path, parameters, nil
}

func validOrigin(origin string) bool {
	projected, _, _, err := projectURL(origin + "/")
	return err == nil && projected == origin
}

func projectedLocations(raw string) bool {
	u, err := url.Parse(raw)
	return err == nil && (u.User != nil || u.RawQuery != "" || u.Fragment != "")
}

func validLabel(name string) bool {
	return len(name) > 0 && len(name) <= MaxTagBytes && validText(name) &&
		!strings.ContainsFunc(name, unicode.IsControl)
}

func projectAnnotation(note string, tags []string, a Authority) (Annotation, error) {
	if len(note) > MaxNoteBytes || len(tags) > MaxTags || !validText(note) {
		return Annotation{}, ErrInvalid
	}
	annotation := Annotation{Provenance: a.provenance(), Note: note, Tags: []string{}}
	for _, tag := range tags {
		if !validLabel(tag) {
			return Annotation{}, ErrInvalid
		}
		annotation.Tags = append(annotation.Tags, tag)
	}
	slices.Sort(annotation.Tags)
	annotation.Tags = slices.Compact(annotation.Tags)
	return annotation, nil
}

func (a Authority) provenance() Provenance {
	return Provenance{SourceID: a.SourceID, Kind: a.Kind, Mode: a.Mode, RunID: a.RunID, Deployment: a.Deployment, CredentialRevision: a.CredentialRevision, SessionGeneration: a.SessionGeneration}
}

func (a Authority) contextKey() string {
	if a.IdentityID == "" {
		if a.SessionGeneration != "" {
			return key("unknown-session", a.Engagement, a.RunID, a.SessionGeneration)
		}
		return key("unknown", a.Engagement, a.RunID)
	}
	if a.Engagement != "" {
		return key("engagement", a.Engagement, a.IdentityID)
	}
	return key("run", a.RunID, a.IdentityID)
}

func (a Authority) validate() error {
	if !contracts.ValidResourceID(a.SourceID) || !contracts.ValidSitemapSourceMode(a.Mode) ||
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
