package scan

import (
	"fmt"
	"sort"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
)

func BenchmarkHTTPRequestSetDecode(b *testing.B) {
	for _, size := range []struct {
		name           string
		requests, body int
	}{{"small", 1, 0}, {"2MiB", 128, 16384}} {
		b.Run(size.name, func(b *testing.B) {
			revision := "rev-1"
			value := HTTPRequestSet{
				SchemaVersion: 1,
				Source: RequestSetSource{
					Artifact:      contracts.ArtifactRef{Namespace: "inputs", Name: "openapi", Revision: &revision},
					ContentDigest: "sha256:" + strings.Repeat("a", 64),
				},
				PreparationDigest: "sha256:" + strings.Repeat("b", 64),
				Gaps:              []PreparationGap{},
				Coverage:          RequestSetCoverage{Operations: size.requests, Prepared: size.requests, Complete: true},
			}
			body := strings.Repeat("x", size.body)
			for i := 0; i < size.requests; i++ {
				request := PreparedHTTPRequest{Method: "POST", URL: fmt.Sprintf("https://example.test/route-%d", i), Headers: []HTTPRequestHeader{}, Body: body}
				digest, err := RequestContentDigest(request)
				if err != nil {
					b.Fatal(err)
				}
				value.Requests = append(value.Requests, RequestSetEntry{
					ID:            "request-" + strings.TrimPrefix(digest, "sha256:"),
					ContentDigest: digest,
					Request:       request,
					Origins:       []RequestOrigin{{Pointer: fmt.Sprintf("#/paths/~1route-%d/post", i)}},
				})
			}
			sort.Slice(value.Requests, func(i, j int) bool { return value.Requests[i].ID < value.Requests[j].ID })
			raw, err := MarshalHTTPRequestSet(value)
			if err != nil {
				b.Fatal(err)
			}
			b.SetBytes(int64(len(raw)))
			b.ReportAllocs()
			for b.Loop() {
				if _, err := DecodeHTTPRequestSet(raw); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
