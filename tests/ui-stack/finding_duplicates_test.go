//go:build e2e

package uistack

import (
	"archive/zip"
	"bytes"
	"context"
	"crypto/ed25519"
	"crypto/x509"
	"encoding/json"
	"encoding/pem"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

// The trusted fixture supplies a source version with pinned Git provenance.
// Actual Runtime publication, Server admission/import and browser reads then
// exercise production paths; the fixture does not create receipts or findings.
func (s *uiStack) prepareCorrelationSource(ctx context.Context, projectID string) (artifacts.ArtifactRef, error) {
	var out artifacts.ArtifactRef
	key := ed25519.NewKeyFromSeed(make([]byte, ed25519.SeedSize))
	der, err := x509.MarshalPKCS8PrivateKey(key)
	if err != nil {
		return out, err
	}
	literal := pem.EncodeToMemory(&pem.Block{Type: "PRIVATE KEY", Bytes: der})
	var archive bytes.Buffer
	writer := zip.NewWriter(&archive)
	member, err := writer.Create("key.py")
	if err != nil {
		return out, err
	}
	if _, err = fmt.Fprintf(member, "key = %q\n", string(literal)); err != nil {
		return out, err
	}
	if err = writer.Close(); err != nil {
		return out, err
	}
	prepared, err := artifacts.PreparePayload(ctx, artifacts.Payload{MediaType: "application/zip", Data: archive.Bytes()})
	if err != nil {
		return out, err
	}
	err = persistencepostgres.InTx(ctx, s.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		repo := artifacts.NewPostgresTxRepository(tx)
		store, err := artifacts.NewService(repo).Project(projectID)
		if err != nil {
			return err
		}
		value, err := store.Write(ctx, artifacts.ArtifactRef{Namespace: "sources", Name: "duplicate-source"}, prepared, nil)
		if err != nil {
			return err
		}
		scope, err := artifacts.ProjectScope(projectID)
		if err != nil {
			return err
		}
		out = value.Ref
		return repo.RecordGitSource(ctx, scope, out, artifacts.GitSource{
			RepositoryURL: "https://example.test:443/retained-source.git", ResolvedCommit: strings.Repeat("a", 40), ImportedAt: time.Now().UTC(),
		})
	})
	return out, err
}

func (s *uiStack) correlationSourceHandler(w http.ResponseWriter, r *http.Request) {
	if r.Header.Get("Authorization") != "Bearer "+s.controlToken || r.ContentLength > 0 {
		http.Error(w, "forbidden", http.StatusForbidden)
		return
	}
	source, err := s.prepareCorrelationSource(r.Context(), r.PathValue("projectId"))
	if err != nil {
		http.Error(w, "prepare correlation fixture: "+err.Error(), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(source); err != nil {
		s.t.Error(err)
	}
}

func correlationProducerStage() gatewayStage {
	return gatewayStage{name: "retained-duplicate-source", tools: []string{
		"open_source_archive", "list_source_files", "read_source", "search_source", "finding", "write_text_artifact",
	}, steps: []gatewayStep{
		toolGatewayStep("open_source_archive", stageRefArguments("source", nil)),
		toolGatewayStep("read_source", fixedArguments(map[string]any{"path": "key.py", "start_line": 1, "max_lines": 4})),
		toolGatewayStep("finding", func(request map[string]any) (map[string]any, error) {
			source, found := stageArtifact(request, "source")
			if !found {
				return nil, fmt.Errorf("selected source input missing")
			}
			return map[string]any{"title": "Embedded private key", "description": "A private key literal is embedded in source.", "file": "key.py", "line": 1, "cwe": "CWE-321", "evidence_refs": []any{source}}, nil
		}),
		toolGatewayStep("write_text_artifact", fixedArguments(map[string]any{"name": "report", "media_type": "text/markdown", "text": "Synthetic key literal recorded with exact source evidence."})),
		{summary: "Source observation recorded", artifacts: map[string]artifactBinding{"report": {namespace: "source-findings", name: "report"}}},
	}}
}
