//go:build e2e

package e2e

import (
	"encoding/json"
	"net/http"
	"sync"
	"testing"
)

func TestEngagementIdentityLifecycleAcrossProcesses(t *testing.T) {
	t.Run("cancel_cookie_request_joins_and_next_allocation_is_fresh", func(t *testing.T) {
		secure := newEngagementTargetFixture(t)
		target := newEngagementHTTPObserver(t, engagementFixtureIP(t, secure))
		h := startEngagementProcessHarnessOptions(t, secure.caFile, "pass", nil, false)
		pool := engagementPool(t, h)
		const session = "private-cancel-session-canary"
		const transient = "private-transient-allocation-canary"
		credential := engagementCredential(t, h, "manual-session", "target-cookies@1", map[string]any{"cookies": map[string]string{"session": session}}, session, transient)
		f := newEngagementIdentityAudit(t, h, []string{target.server.URL}, []string{target.server.URL + "/prime", target.server.URL + "/resource"})
		f.addIdentity(t, "account-a", "cookies", credential)
		entered, closed := make(chan struct{}), make(chan struct{})
		var mu sync.Mutex
		resources, primes, violations := 0, 0, 0
		target.setHandler(func(w http.ResponseWriter, r *http.Request) {
			if engagementCookie(r, "session") != session {
				w.WriteHeader(http.StatusOK)
				return
			}
			mu.Lock()
			if r.URL.Path == "/prime" {
				primes++
				if engagementCookie(r, "transient") != "" {
					violations++
				}
				mu.Unlock()
				http.SetCookie(w, &http.Cookie{Name: "transient", Value: transient, Path: "/"})
				w.WriteHeader(http.StatusOK)
				return
			}
			resources++
			first := resources == 1
			if engagementCookie(r, "transient") != transient {
				violations++
			}
			mu.Unlock()
			if first {
				close(entered)
				<-r.Context().Done()
				close(closed)
				return
			}
			w.WriteHeader(http.StatusOK)
		})
		f.requests = []map[string]any{{"url": target.server.URL + "/prime", "credential_id": credential.CredentialID}, {"url": target.server.URL + "/resource", "credential_id": credential.CredentialID}}
		audit := f.start(t)
		select {
		case <-entered:
		case <-h.ctx.Done():
			t.Fatal("cookie request did not reach target")
		}
		var allocation, instance string
		if err := pool.QueryRow(h.ctx, `SELECT p.allocation_id,p.runtime_instance_id FROM audit_action_attempts p JOIN audit_actions a USING(audit_id,action_id) WHERE p.audit_id=$1 AND a.identity_id='account-a' AND a.phase='request' AND a.step_index=1 AND p.state='started'`, audit).Scan(&allocation, &instance); err != nil {
			t.Fatal(err)
		}
		var revision uint64
		if err := pool.QueryRow(h.ctx, `SELECT revision FROM audits WHERE audit_id=$1`, audit).Scan(&revision); err != nil {
			t.Fatal(err)
		}
		engagementAuditPOST(t, h, "/v1/audits/"+audit+"/cancel", nil, revision, http.StatusAccepted, nil)
		select {
		case <-closed:
		case <-h.ctx.Done():
			t.Fatal("cancel did not join cookie request")
		}
		waitEngagementCondition(t, h.ctx, func() bool {
			var released bool
			if err := pool.QueryRow(h.ctx, `SELECT release_completed_at IS NOT NULL FROM stage_allocations WHERE allocation_id=$1`, allocation).Scan(&released); err != nil {
				t.Fatal(err)
			}
			return released
		})
		next := f.start(t)
		waitForAuditProgram(t, h.ctx, h.server, h.runtimeProcess, h.gateway, h.client, h.baseURL, next, 1)
		mu.Lock()
		if primes != 2 || resources != 2 || violations != 0 {
			t.Errorf("replacement reused cookies: primes=%d resources=%d violations=%d", primes, resources, violations)
		}
		mu.Unlock()
		h.gateway.releaseBlockedRequest()
		waitForAuditProgram(t, h.ctx, h.server, h.runtimeProcess, h.gateway, h.client, h.baseURL, next, 1)
		assertEngagementSecretsAbsent(t, h, pool)
	})
	t.Run("credential_kind_mismatch_is_rejected_before_target_traffic", func(t *testing.T) {
		secure := newEngagementTargetFixture(t)
		target := newEngagementHTTPObserver(t, engagementFixtureIP(t, secure))
		h := startEngagementProcessHarness(t, secure.caFile, "pass")
		pool := engagementPool(t, h)
		const secret = "private-rejected-session-canary"
		credential := engagementCredential(t, h, "ready-session", "target-cookies@1", map[string]any{"cookies": map[string]string{"session": secret}}, secret)
		f := newEngagementIdentityAudit(t, h, []string{target.server.URL}, []string{target.server.URL + "/resource"})
		f.addIdentity(t, "account-a", "cookies", credential)
		encoded, err := json.Marshal(f.document)
		if err != nil {
			t.Fatal(err)
		}
		var document map[string]any
		if err = json.Unmarshal(encoded, &document); err != nil {
			t.Fatal(err)
		}
		document["identities"].([]any)[1].(map[string]any)["credential"].(map[string]any)["kind"] = "target-bearer@1"
		encoded, err = json.Marshal(document)
		if err != nil {
			t.Fatal(err)
		}
		ref := uploadProjectScopeArtifact(t, h.client, h.baseURL, f.project.ProjectID, "inputs", "unsupported-engagement", "application/vnd.contractor.engagement+json", encoded)
		f.engagement = &ref
		engagementAuditPOST(t, h, "/v1/projects/"+f.project.ProjectID+"/audits", map[string]any{
			"profile": map[string]string{"name": "engagement-fixture", "version": "1"},
			"inputs":  map[string]artifactRef{"checklist": f.checklist, "engagement": ref},
			"scope":   map[string]string{"objective": "Verify private identity and session boundaries"},
		}, 0, http.StatusBadRequest, nil)
		if target.bytes.Load() != 0 || target.connections.Load() != 0 {
			t.Fatal("mismatched credential reached target transport")
		}
		assertEngagementSecretsAbsent(t, h, pool)
	})
}
