//go:build e2e

package e2e

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/engagementcontract"
	"github.com/grauwolf32/contractor/internal/contracts/runtimesettings"
	"github.com/jackc/pgx/v5/pgxpool"
)

type engagementIdentityAudit struct {
	h          *engagementProcessHarness
	project    projectResourceResponse
	checklist  artifactRef
	engagement *artifactRef
	document   engagementcontract.Document
	requests   []map[string]any
}

func newEngagementIdentityAudit(t *testing.T, h *engagementProcessHarness, origins, seeds []string) *engagementIdentityAudit {
	t.Helper()
	f := &engagementIdentityAudit{h: h}
	for _, url := range seeds {
		f.requests = append(f.requests, map[string]any{"url": url})
	}
	engagementAuditPOST(t, h, "/v1/projects", map[string]string{"kind": "project", "name": "Private identity fixture", "activeChecks": "automatic"}, 0, http.StatusCreated, &f.project)
	checks := []byte(`{"schema":"contractor.audit.checklist.v1","items":[{"key":"check-preflight","version":"1","statement":"Record private identity conformance.","applicability":"always","allowed_methods":["static"],"required_evidence":[],"review_policy":"automatic"}]}`)
	f.checklist = uploadProjectScopeArtifact(t, h.client, h.baseURL, f.project.ProjectID, "inputs", "checks", "application/json", checks)
	routes := make([]engagementcontract.Route, 0, len(origins))
	for _, origin := range origins {
		route := engagementRoute(origin, "/")
		route.Methods = []string{"GET", "POST"}
		routes = append(routes, route)
	}
	f.document = engagementcontract.Document{
		Schema: engagementcontract.Schema, Target: engagementcontract.Target{Origins: origins},
		Scope:      engagementcontract.Scope{Routes: routes, Exclusions: []engagementcontract.Route{}},
		Identities: []engagementcontract.Identity{{ID: "anonymous", Kind: "anonymous", Origins: origins}},

		Policy:  engagementcontract.Policy{Version: "1", ActiveChecks: "automatic", RequiredCapabilities: []string{engagementcontract.Capability, "http-target@1"}, OptionalCapabilities: []string{}},
		Budgets: engagementcontract.HTTPLimits{},
	}
	return f
}

func engagementCredential(t *testing.T, h *engagementProcessHarness, id, kind string, material any, canaries ...string) *runtimesettings.RuntimeCredentialRef {
	t.Helper()
	h.protectSecrets(canaries...)
	var metadata struct {
		CredentialID string `json:"credentialId"`
		Kind         string `json:"kind"`
	}
	body, err := json.Marshal(map[string]any{"credentialId": id, "kind": kind, "material": material})
	if err != nil {
		t.Fatal(err)
	}
	request, err := http.NewRequestWithContext(h.ctx, http.MethodPost, h.baseURL+"/v1/operations/runtime-credentials", bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Authorization", "Bearer "+publicToken)
	request.Header.Set("Idempotency-Key", "credential-"+randomHex(t, 8))
	response, err := h.client.Do(request)
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	encoded, err := io.ReadAll(io.LimitReader(response.Body, 16385))
	if err != nil || len(encoded) > 16384 || response.StatusCode != http.StatusCreated || bytes.Contains(encoded, []byte(`"material"`)) {
		t.Fatal("credential creation did not return bounded write-only metadata")
	}
	for _, canary := range canaries {
		if bytes.Contains(encoded, []byte(canary)) {
			t.Fatal("credential create response exposed a private canary")
		}
	}
	if err := json.Unmarshal(encoded, &metadata); err != nil {
		t.Fatal(err)
	}
	if metadata.CredentialID != id || metadata.Kind != kind {
		t.Fatal("credential create did not return the exact non-secret identity")
	}
	return &runtimesettings.RuntimeCredentialRef{CredentialID: id, Kind: runtimesettings.RuntimeCredentialKind(kind)}
}

func (f *engagementIdentityAudit) addIdentity(t *testing.T, id, kind string, credential *runtimesettings.RuntimeCredentialRef) {
	t.Helper()
	f.document.Identities = append(f.document.Identities, engagementcontract.Identity{ID: id, Kind: kind, Credential: credential, Origins: f.document.Target.Origins})
}

func (f *engagementIdentityAudit) draft(t *testing.T) auditProgramAudit {
	t.Helper()
	if err := f.document.Validate(); err != nil {
		t.Fatal(err)
	}
	body, err := json.Marshal(f.document)
	if err != nil {
		t.Fatal(err)
	}
	if f.engagement == nil {
		ref := uploadProjectScopeArtifact(t, f.h.client, f.h.baseURL, f.project.ProjectID, "inputs", "engagement", engagementcontract.MediaType, body)
		f.engagement = &ref
	}
	var draft auditProgramAudit
	engagementAuditPOST(t, f.h, "/v1/projects/"+f.project.ProjectID+"/audits", map[string]any{"profile": map[string]string{"name": "engagement-fixture", "version": "1"}, "inputs": map[string]artifactRef{"checklist": f.checklist, "engagement": *f.engagement}, "scope": map[string]string{"objective": "Verify private identity and session boundaries"}}, 0, http.StatusCreated, &draft)
	return draft
}

func (f *engagementIdentityAudit) start(t *testing.T) string {
	t.Helper()
	setEngagementRequests(f.h, f.requests)
	draft := f.draft(t)
	engagementAuditPOST(t, f.h, "/v1/audits/"+draft.AuditID+"/start", nil, draft.Revision, http.StatusOK, nil)
	return draft.AuditID
}

// Read raw process logs, retained rows and artifact bytes before any diagnostic
// redaction. All queries address this harness's disposable, bounded database.
func assertEngagementSecretsAbsent(t *testing.T, h *engagementProcessHarness, pool *pgxpool.Pool, evidence ...[]byte) {
	t.Helper()
	h.secretsMu.Lock()
	secrets := append([]string(nil), h.secrets...)
	h.secretsMu.Unlock()
	check := func(surface string, data []byte) {
		for _, secret := range secrets {
			if bytes.Contains(data, []byte(secret)) {
				t.Fatalf("private identity canary or its raw digest escaped into %s", surface)
			}
		}
	}
	for _, body := range evidence {
		check("preflight/public response", body)
	}
	for _, process := range []*childProcess{h.server, h.runtimeProcess} {
		if process.logs.canaryLeaked() {
			t.Fatalf("private identity canary escaped into %s before log truncation/redaction", process.name)
		}
		check(process.name+" raw logs", []byte(process.logs.redacted()))
	}
	for _, table := range []string{"projects", "audits", "workflow_runs", "stage_allocations", "workflow_run_events", "planner_sessions", "planner_execution_reports", "allocation_execution_reports", "stage_metrics", "runtime_credentials"} {
		rows, err := pool.Query(h.ctx, "SELECT to_jsonb(t)::text FROM "+table+" t LIMIT 4097")
		if err != nil {
			t.Fatalf("read retained %s: %v", table, err)
		}
		count := 0
		for rows.Next() {
			var body string
			if err := rows.Scan(&body); err != nil {
				rows.Close()
				t.Fatal(err)
			}
			count++
			check(table, []byte(body))
		}
		err = rows.Err()
		rows.Close()
		if err != nil || count > 4096 {
			t.Fatalf("retained %s scan failed or exceeded fixture bound: %v", table, err)
		}
	}
	rows, err := pool.Query(h.ctx, "SELECT payload FROM artifact_blobs LIMIT 257")
	if err != nil {
		t.Fatal(err)
	}
	defer rows.Close()
	count := 0
	for rows.Next() {
		var payload []byte
		if err := rows.Scan(&payload); err != nil {
			t.Fatal(err)
		}
		count++
		check("artifact payload", payload)
	}
	if err := rows.Err(); err != nil || count > 256 {
		t.Fatalf("artifact scan failed or exceeded fixture bound: %v", err)
	}
	if failures := h.gateway.Failures(); len(failures) != 0 {
		t.Fatal("Gateway rejected private material or unexpected Worker traffic")
	}
}

func engagementCookie(r *http.Request, name string) string {
	cookie, err := r.Cookie(name)
	if err != nil {
		return ""
	}
	return cookie.Value
}
