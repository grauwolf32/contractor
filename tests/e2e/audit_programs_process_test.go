//go:build e2e

package e2e

import (
	"archive/zip"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"sort"
	"testing"
	"time"

	artifactpolicy "github.com/grauwolf32/contractor/internal/artifacts/policy"
	"github.com/grauwolf32/contractor/internal/auditstandards"
	"github.com/jackc/pgx/v5/pgxpool"
)

type auditProgramAudit struct {
	AuditID           string `json:"auditId"`
	ProjectID         string `json:"projectId"`
	State             string `json:"state"`
	Revision          uint64 `json:"revision"`
	SubmittedRunCount int    `json:"submittedRunCount"`
	OutstandingRuns   int    `json:"outstandingRunCount"`
	Baseline          *struct {
		Skills []struct {
			Name         string      `json:"name"`
			Source       artifactRef `json:"source"`
			SourceDigest string      `json:"sourceDigest"`
			SourceSize   int64       `json:"sourceSize"`
		} `json:"skills"`
		Standards []struct {
			Reference struct {
				Scheme  string `json:"scheme"`
				Version string `json:"version"`
			} `json:"reference"`
			Source struct {
				Revision string `json:"revision"`
			} `json:"source"`
			License struct {
				ID string `json:"id"`
			} `json:"license"`
			Catalog  auditProgramExactPackage `json:"catalog"`
			Retained auditProgramExactPackage `json:"retained"`
		} `json:"standards"`
	} `json:"baseline,omitempty"`
}

type auditProgramExactPackage struct {
	Artifact artifactRef `json:"artifact"`
	Digest   string      `json:"digest"`
}

func waitForAuditProgram(
	t *testing.T,
	ctx context.Context,
	server, runtimeProcess *childProcess,
	gateway *domainGateway,
	client *http.Client,
	baseURL, auditID string,
	expectedRuns int,
) auditProgramAudit {
	t.Helper()
	ticker := time.NewTicker(150 * time.Millisecond)
	defer ticker.Stop()
	var lastAudit auditProgramAudit
	for {
		var audit auditProgramAudit
		if auditProgramTryGET(ctx, client, baseURL+"/v1/audits/"+url.PathEscape(auditID), &audit) == nil {
			lastAudit = audit
			switch audit.State {
			case "completed":
				var page struct {
					Tasks []basicTask `json:"items"`
					Next  *int        `json:"nextOrdinal"`
				}
				if err := auditProgramTryGET(ctx, client, baseURL+"/v1/audits/"+url.PathEscape(auditID)+"/tasks?limit=128", &page); err != nil {
					break
				}
				seen := map[string]bool{}
				for _, task := range page.Tasks {
					if task.Attempts != 1 || task.LastExecution == nil || task.State != "settled" || task.GapCode != nil || task.Result == nil {
						t.Fatalf("common task failed: %+v", task)
					}
					seen[*task.LastExecution] = true
				}
				if audit.SubmittedRunCount != expectedRuns || audit.OutstandingRuns != 0 || page.Next != nil || len(seen) != expectedRuns {
					t.Fatalf("completed Audit counters or attempts differ: %+v tasks=%+v", audit, page.Tasks)
				}
				t.Logf("Audit %s completed with exactly %d Worker attempts", auditID, expectedRuns)
				return audit
			case "failed", "cancelled":
				t.Fatalf("Audit reached %s: %+v\nserver:\n%s\nruntime:\n%s\ngateway: %v",
					audit.State, audit, server.logs.redacted(publicToken, llmGatewayToken),
					runtimeProcess.logs.redacted(publicToken, llmGatewayToken), gateway.Failures())
			}
		}
		for _, process := range []*childProcess{server, runtimeProcess} {
			if exited, processErr := process.exited(); exited {
				t.Fatalf("%s exited while Audit was active: %v\n%s", process.name, processErr,
					process.logs.redacted(publicToken, llmGatewayToken))
			}
		}
		select {
		case <-ctx.Done():
			t.Fatalf("wait for Audit: %v; last state=%s submitted=%d outstanding=%d; gateway stages=%d failures=%v\nserver:\n%s\nruntime:\n%s", ctx.Err(),
				lastAudit.State, lastAudit.SubmittedRunCount, lastAudit.OutstandingRuns,
				gateway.CompletedStages(), gateway.Failures(),
				server.logs.redacted(publicToken, llmGatewayToken),
				runtimeProcess.logs.redacted(publicToken, llmGatewayToken))
		case <-ticker.C:
		}
	}
}

func deleteASVSBacktraceRun(
	t *testing.T, ctx context.Context, client *http.Client, baseURL, runID string,
) {
	t.Helper()
	waitForAuditRunDeletable(t, ctx, client, baseURL, runID)
	deleteRequest, err := http.NewRequest(http.MethodDelete, baseURL+"/v1/runs/"+url.PathEscape(runID), nil)
	if err != nil {
		t.Fatal(err)
	}
	deleteRequest.Header.Set("Authorization", "Bearer "+publicToken)
	deleteResponse := do(t, client, deleteRequest, http.StatusNoContent)
	deleteResponse.Body.Close()
}

func replaceAuditProgramCatalog(
	t *testing.T, ctx context.Context, databaseURL, ownerID, configRoot string,
) {
	t.Helper()
	removeAuditProgramAuthoringEntries(t, configRoot)
	pool, err := pgxpool.New(ctx, databaseURL)
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Close()
	for _, reference := range []auditstandards.Reference{
		{Scheme: "owasp-asvs", Version: "5.0.0"},
		{Scheme: "owasp-web-top10", Version: "2025"},
	} {
		result, deleteErr := pool.Exec(ctx, `
DELETE FROM artifact_bindings
 WHERE scope_kind = 'user' AND scope_id = $1
   AND namespace = $2 AND name = $3`,
			ownerID, artifactpolicy.AuditStandardCatalogNamespace, auditstandards.ArtifactName(reference),
		)
		if deleteErr != nil {
			t.Fatalf("replace current %s@%s catalog binding: %v", reference.Scheme, reference.Version, deleteErr)
		}
		if result.RowsAffected() != 1 {
			t.Fatalf("replaced %s@%s catalog bindings = %d, want 1",
				reference.Scheme, reference.Version, result.RowsAffected())
		}
	}
}

var removedAuditProgramProfiles = map[string]bool{
	"openapi-operation-observe@1":      true,
	"openapi-operation-trace@2":        true,
	"owasp-asvs-5-0-l1-source-pilot@2": true,
	"owasp-top10-2025-source-risk@2":   true,
}

func removeAuditProgramAuthoringEntries(t *testing.T, configRoot string) {
	t.Helper()
	// Remove the current Memory program definitions with their standard sources.
	// Leave the separate planner instruction files intact.
	for _, relative := range []string{
		"audit-profiles/openapi_operation_observe.yaml",
		"audit-profiles/openapi_operation_trace.yaml",
		"audit-profiles/owasp_asvs_5_0_l1_source_pilot.yaml",
		"audit-profiles/owasp_top10_2025_source_risk.yaml",
		"audit-standards/owasp-asvs-5.0.0",
		"audit-standards/owasp-web-top10-2025",
		"agent-templates/audit_asvs_source_verifier.yaml",
		"agent-templates/audit_risk_source_checker.yaml",
		"instructions/audit-asvs-source-verifier-worker.md",
		"instructions/audit-risk-source-checker-worker.md",
		"workflows/audit_asvs_source_verification.yaml",
		"workflows/audit_top10_source_risk.yaml",
	} {
		if err := os.RemoveAll(filepath.Join(configRoot, filepath.FromSlash(relative))); err != nil {
			t.Fatalf("remove staged Audit program catalog entry %s: %v", relative, err)
		}
	}
}

func assertAuditProgramCatalogUnavailable(t *testing.T, client *http.Client, baseURL string) {
	t.Helper()
	var page struct {
		Items []struct {
			Ref struct {
				Name    string `json:"name"`
				Version string `json:"version"`
			} `json:"ref"`
		} `json:"items"`
	}
	auditProgramGET(t, client, baseURL+"/v1/audit-profiles?limit=100", &page)
	remainingProfiles := map[string]bool{
		"source-checklist@3":                false,
		"owasp-asvs-5-0-l1-source-review@2": false,
	}
	for _, profile := range page.Items {
		identity := profile.Ref.Name + "@" + profile.Ref.Version
		if removedAuditProgramProfiles[identity] {
			t.Fatalf("replaced Audit program profile remains current: %+v", profile)
		}
		if _, required := remainingProfiles[identity]; required {
			remainingProfiles[identity] = true
		}
	}
	for selector, found := range remainingProfiles {
		if !found {
			t.Fatalf("unrelated Audit program profile is unavailable: %s", selector)
		}
	}
	for _, path := range []string{
		"/v1/audit-standards/owasp-asvs/versions/5.0.0",
		"/v1/audit-standards/owasp-web-top10/versions/2025",
	} {
		request, err := http.NewRequest(http.MethodGet, baseURL+path, nil)
		if err != nil {
			t.Fatal(err)
		}
		request.Header.Set("Authorization", "Bearer "+publicToken)
		response := do(t, client, request, http.StatusNotFound)
		response.Body.Close()
	}
}

func waitForAuditRunDeletable(
	t *testing.T, ctx context.Context, client *http.Client, baseURL, runID string,
) {
	t.Helper()
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()
	for {
		var run runStatus
		if auditProgramTryGET(ctx, client, baseURL+"/v1/runs/"+url.PathEscape(runID), &run) == nil && run.Deletable {
			return
		}
		select {
		case <-ctx.Done():
			t.Fatalf("wait for Audit child Run %s deletion: %v", runID, ctx.Err())
		case <-ticker.C:
		}
	}
}

func auditProgramGET(t *testing.T, client *http.Client, target string, output any) {
	t.Helper()
	if err := auditProgramTryGET(context.Background(), client, target, output); err != nil {
		t.Fatal(err)
	}
}

func decodeAuditProgramResponse(t *testing.T, response *http.Response, output any) {
	t.Helper()
	if err := json.NewDecoder(response.Body).Decode(output); err != nil {
		t.Fatalf("decode HTTP %d response: %v", response.StatusCode, err)
	}
}

func auditProgramTryGET(ctx context.Context, client *http.Client, target string, output any) error {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, target, nil)
	if err != nil {
		return err
	}
	request.Header.Set("Authorization", "Bearer "+publicToken)
	response, err := client.Do(request)
	if err != nil {
		return err
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return fmt.Errorf("GET %s returned HTTP %d", target, response.StatusCode)
	}
	if err := json.NewDecoder(response.Body).Decode(output); err != nil {
		return fmt.Errorf("decode GET %s: %w", target, err)
	}
	return nil
}

func auditProgramZip(t *testing.T, files map[string][]byte) []byte {
	t.Helper()
	names := make([]string, 0, len(files))
	for name := range files {
		names = append(names, name)
	}
	sort.Strings(names)
	var output bytes.Buffer
	writer := zip.NewWriter(&output)
	for _, name := range names {
		header := &zip.FileHeader{Name: name, Method: zip.Store}
		header.SetMode(0o600)
		header.SetModTime(time.Date(2026, 9, 6, 0, 0, 0, 0, time.UTC))
		entry, err := writer.CreateHeader(header)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := entry.Write(files[name]); err != nil {
			t.Fatal(err)
		}
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	return output.Bytes()
}
