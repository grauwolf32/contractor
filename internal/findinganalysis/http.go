package findinganalysis

import (
	"bufio"
	"bytes"
	"encoding/base64"
	"mime"
	"net/http"
	"net/url"
	"strings"

	artifactpolicy "github.com/grauwolf32/contractor/internal/artifacts/policy"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/scanplan"
)

type ResolvedHTTP struct {
	ID          string   `json:"id"`
	Method      string   `json:"method,omitempty"`
	URI         string   `json:"uri,omitempty"`
	Status      *int     `json:"status,omitempty"`
	ContentType string   `json:"content_type,omitempty"`
	Omitted     []string `json:"omitted"`
	// Evidence origin remains producer-supplied; parsing a Server envelope does
	// not prove an uploaded artifact was actually minted by trusted capture.
	Origin string `json:"origin"`
}

// ProjectHTTP decodes one exact selected exchange without accessing its source.
func ProjectHTTP(exchange HTTPExchange, doc auditdomain.FindingCollectionDocument, data []byte) (ResolvedHTTP, error) {
	result := ResolvedHTTP{ID: exchange.ID, Origin: "producer-asserted", Omitted: []string{"bodies", "headers", "query", "fragment", "userinfo"}}
	if exchange.Decoder == "http-response-body/v1" {
		// A body carries no method, request URI, status or session identity.
		result.ContentType, _, _ = mime.ParseMediaType(doc.MediaType)
		return result, nil
	}
	if exchange.Decoder != "contractor-http-exchange/v1" || doc.MediaType != auditdomain.HTTPExchangeMediaType {
		return result, invalidEvidence()
	}
	envelope, err := auditdomain.DecodeHTTPExchangeArtifact(data)
	if err != nil {
		return result, invalidEvidence()
	}

	// Only the explicitly selected exchange is used; its final attempt is the
	// available outcome. Previous attempts stay in the native access-controlled
	// evidence and do not become fabricated independent requests.
	attempt := envelope.Capture.Attempts[len(envelope.Capture.Attempts)-1]
	parsed, err := url.Parse(attempt.Request.URL)
	if err != nil {
		return result, invalidEvidence()
	}
	parsed.User = nil
	parsed.RawQuery = ""
	parsed.ForceQuery = false
	parsed.Fragment = ""
	parsed.RawFragment = ""
	result.Method = attempt.Request.Method
	result.URI = parsed.String()
	if response := attempt.Response; response != nil {
		status := response.Status
		result.Status = &status
		if response.Message.Kind == "structured" {
			for _, h := range *response.Message.Headers {
				if strings.EqualFold(h.Name, "content-type") {
					result.ContentType = h.Value
					break
				}
			}
		} else {
			raw, err := base64.StdEncoding.Strict().DecodeString(*response.Message.DataBase64)
			if err != nil {
				return result, invalidEvidence()
			}
			response, err := http.ReadResponse(bufio.NewReader(bytes.NewReader(raw)), nil)
			if err != nil || response.StatusCode != status {
				return result, invalidEvidence()
			}
			result.ContentType = response.Header.Get("Content-Type")
			// The body is never read, copied, or embedded in the export.
			if response.Body != nil {
				_ = response.Body.Close()
			}
		}
	}
	if value, _, err := mime.ParseMediaType(result.ContentType); err == nil {
		result.ContentType = value
	} else {
		result.ContentType = ""
	}
	if doc.Scope.Kind == "run" && doc.Ref.Namespace == artifactpolicy.HTTPExchangeNamespace && doc.Scope.ID == envelope.Origin.RunID && doc.Ref.Name == auditdomain.HTTPExchangeArtifactName(envelope.Origin.InvocationID, envelope.Capture.Source, envelope.Capture.RequestID) {
		result.Origin = "runtime-capture"
	}
	return result, nil
}

func resolveOperation(operation Operation, evidence evidenceSet) error {
	doc, data, ok := evidence.get(operation.DocumentRef)
	if !ok {
		return invalidEvidence()
	}
	root, err := scanplan.ParseDocument(data, doc.MediaType)
	if err != nil {
		return invalidEvidence()
	}
	version, ok := root["openapi"].(string)
	if !ok || !strings.HasPrefix(version, "3.") {
		return invalidEvidence()
	}
	// An exact two-component paths operation pointer proves method/path binding;
	// resolving arbitrary refs or operationId searches would invent a join.
	parts := strings.Split(operation.Pointer, "/")
	if len(parts) != 4 || parts[0] != "" || parts[1] != "paths" {
		return invalidEvidence()
	}
	unescape := func(s string) (string, bool) {
		for i := 0; i < len(s); i++ {
			if s[i] == '~' {
				if i+1 >= len(s) || s[i+1] != '0' && s[i+1] != '1' {
					return "", false
				}
				i++
			}
		}
		return strings.ReplaceAll(strings.ReplaceAll(s, "~1", "/"), "~0", "~"), true
	}
	path, ok := unescape(parts[2])
	if !ok || path != operation.PathTemplate || parts[3] != strings.ToLower(operation.Method) {
		return invalidEvidence()
	}
	if !strings.Contains(" get put post delete options head patch trace ", " "+parts[3]+" ") {
		return invalidEvidence()
	}
	paths, ok := root["paths"].(map[string]any)
	if !ok {
		return invalidEvidence()
	}
	item, ok := paths[path].(map[string]any)
	if !ok {
		return invalidEvidence()
	}
	if _, ok := item[parts[3]].(map[string]any); !ok {
		return invalidEvidence()
	}
	return nil
}
