package cli

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"testing"

	"github.com/grauwolf32/contractor/internal/publicclient"
	publicapi "github.com/grauwolf32/contractor/internal/publicclient/generated"
)

type sweepFixture struct {
	t                  *testing.T
	mu                 sync.Mutex
	count, gets, posts int
	changed            bool
	fail               string
	stop               context.CancelFunc
	requests           map[string]string
	runs               map[string]string
	retryProject       string
	missingArtifact    string
}

func (f *sweepFixture) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	f.mu.Lock()
	defer f.mu.Unlock()
	w.Header().Set(publicclient.APIVersionHeader, publicclient.APIVersion)
	w.Header().Set("Content-Type", "application/json")
	if r.Method == http.MethodGet {
		f.gets++
		if f.changed {
			f.t.Error("resume reread a mutable Project or Artifact")
		}
		if r.URL.Path == "/v1/projects" {
			start, _ := strconv.Atoi(r.URL.Query().Get("cursor"))
			end := min(start+113, f.count)
			page := publicapi.ProjectPage{Items: []publicapi.Project{}}
			for i := start; i < end; i++ {
				page.Items = append(page.Items, f.project(i))
			}
			page.Page.HasMore = end < f.count
			if page.Page.HasMore {
				next := strconv.Itoa(end)
				page.Page.NextCursor = &next
			}
			_ = json.NewEncoder(w).Encode(page)
			return
		}
		parts := strings.Split(strings.Trim(r.URL.Path, "/"), "/")
		index, err := strconv.Atoi(strings.TrimPrefix(parts[2], "p"))
		if err != nil {
			f.t.Error(err)
			http.Error(w, "bad Project", 400)
			return
		}
		if len(parts) == 3 {
			_ = json.NewEncoder(w).Encode(f.project(index))
			return
		}
		if strings.HasSuffix(r.URL.Path, "/metadata") {
			if parts[2] == f.missingArtifact {
				w.WriteHeader(404)
				_, _ = io.WriteString(w, `{"code":"not_found","message":"missing","retryable":false}`)
				return
			}
			_ = json.NewEncoder(w).Encode(map[string]any{"artifact": map[string]string{"namespace": "scan", "name": "input", "revision": "revision-1"}, "mediaType": "application/json", "size": 0, "createdAt": "2026-10-23T00:00:00Z", "current": true, "frozen": false})
			return
		}
		f.t.Errorf("unexpected GET %s", r.URL.Path)
		w.WriteHeader(404)
		return
	}
	if r.Method != http.MethodPost {
		f.t.Errorf("unexpected method %s", r.Method)
		w.WriteHeader(405)
		return
	}
	f.posts++
	project := strings.Split(r.URL.Path, "/")[3]
	key := r.Header.Get("Idempotency-Key")
	data, err := io.ReadAll(r.Body)
	if err != nil {
		f.t.Error(err)
	}
	if key == "" {
		f.t.Error("unkeyed sweep create")
	}
	if before, found := f.requests[key]; found && before != string(data) {
		f.t.Error("changed replay request")
	}
	f.requests[key] = string(data)
	if project == f.retryProject {
		w.Header().Set("Retry-After", "0")
		w.WriteHeader(503)
		_, _ = io.WriteString(w, `{"code":"storage_transaction_conflict","message":"retry","retryable":true}`)
		return
	}
	if project == f.fail {
		w.WriteHeader(400)
		_, _ = io.WriteString(w, `{"code":"invalid_request","message":"bad member","retryable":false}`)
		return
	}
	run := f.runs[key]
	if run == "" {
		run = "run-" + project
		f.runs[key] = run
	}
	if project == "p0001" && f.stop != nil {
		stop := f.stop
		f.stop = nil
		stop()
		connection, _, err := w.(http.Hijacker).Hijack()
		if err != nil {
			f.t.Error(err)
		} else {
			connection.Close()
		}
		return
	}
	w.WriteHeader(http.StatusAccepted)
	_ = json.NewEncoder(w).Encode(publicapi.CreateRunResponse{RunId: run, ProjectId: &project, State: "queued"})
}
func (f *sweepFixture) project(index int) publicapi.Project {
	id := fmt.Sprintf("p%04d", index)
	project := publicapi.Project{ProjectId: id, Name: id, Kind: "project", Lifecycle: "active", Description: "application"}
	if index == f.count-1 {
		project.Lifecycle = "deleting"
		return project
	}
	if index == f.count-2 {
		return project
	}
	project.HttpTarget = &publicapi.ProjectHTTPTarget{Url: "http://old.example/" + id}
	return project
}
func newSweepFixture(t *testing.T, count int) (*sweepFixture, *httptest.Server) {
	f := &sweepFixture{t: t, count: count, requests: map[string]string{}, runs: map[string]string{}}
	server := httptest.NewServer(f)
	t.Cleanup(server.Close)
	return f, server
}
func invokeSweep(t *testing.T, ctx context.Context, server, path, action string, extra ...string) error {
	t.Helper()
	var out, stderr bytes.Buffer
	command := New(strings.NewReader(""), &out, &stderr, func(name string) string {
		if name == "CONTRACTOR_API_TOKEN" {
			return "fixture-token"
		}
		return ""
	})
	args := []string{"--server", server, "scan-sweep", action, path}
	args = append(args, extra...)
	return command.Run(ctx, args)
}

func TestScanSweepPagesAndBoundsAtRealisticSize(t *testing.T) {
	for _, count := range []int{7, 1000, 1001} {
		t.Run(strconv.Itoa(count), func(t *testing.T) {
			fixture, server := newSweepFixture(t, count)
			path := filepath.Join(t.TempDir(), "sweep.json")
			err := invokeSweep(t, t.Context(), server.URL, path, "create", "--all", "--param", "template_ids=CVE-fixture")
			if count > maxSweepProjects {
				if err == nil || fixture.posts != 0 {
					t.Fatal("oversized selection dispatched")
				}
				if _, err := os.Stat(path); !errors.Is(err, os.ErrNotExist) {
					t.Fatal("oversized selection saved")
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			manifest, err := loadScanSweep(path)
			if err != nil {
				t.Fatal(err)
			}
			if len(manifest.Outcomes) != count || len(fixture.runs) != count-2 {
				t.Fatalf("members/runs=%d/%d", len(manifest.Outcomes), len(fixture.runs))
			}
			if fixture.gets != (count+112)/113 {
				t.Fatalf("not every page followed: %d", fixture.gets)
			}
			info, err := os.Stat(path)
			if err != nil || info.Mode().Perm() != 0600 || info.Size() > maxSweepManifestBytes {
				t.Fatal("unsafe or unbounded manifest")
			}
			before := fixture.posts
			if err := invokeSweep(t, t.Context(), server.URL, path, "resume"); err != nil {
				t.Fatal(err)
			}
			if fixture.posts != before {
				t.Fatal("settled members submitted again")
			}
		})
	}
}

func TestScanSweepLostAcknowledgementResumesCapturedTargetsAndExactInputs(t *testing.T) {
	fixture, server := newSweepFixture(t, 7)
	path := filepath.Join(t.TempDir(), "sweep.json")
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	fixture.stop = cancel
	err := invokeSweep(t, ctx, server.URL, path, "create", "--all", "--artifact", "request=scan/input", "--param", "template_ids=fixture-id")
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("interrupted sweep=%v", err)
	}
	manifest, err := loadScanSweep(path)
	if err != nil {
		t.Fatal(err)
	}
	if manifest.Outcomes[0].State != "created" || manifest.Outcomes[1].State != "unknown" || manifest.Outcomes[2].State != "pending" {
		t.Fatalf("partial membership=%+v", manifest.Outcomes)
	}
	fixture.changed = true
	if err := invokeSweep(t, t.Context(), server.URL, path, "resume"); err != nil {
		t.Fatal(err)
	}
	if len(fixture.runs) != 5 || fixture.posts != 6 {
		t.Fatalf("lost ack duplicated or omitted Runs: %d/%d", len(fixture.runs), fixture.posts)
	}
	for _, request := range fixture.requests {
		if !strings.Contains(request, "http://old.example/") || !strings.Contains(request, "revision-1") || !strings.Contains(request, "fixture-id") {
			t.Fatal("captured input changed")
		}
	}
}

func TestScanSweepPermanentMemberFailureDoesNotBlockOthers(t *testing.T) {
	fixture, server := newSweepFixture(t, 7)
	fixture.fail = "p0002"
	path := filepath.Join(t.TempDir(), "sweep.json")
	if err := invokeSweep(t, t.Context(), server.URL, path, "create", "--all"); err == nil {
		t.Fatal("permanent member failure did not produce a nonzero result")
	}
	manifest, err := loadScanSweep(path)
	if err != nil {
		t.Fatal(err)
	}
	if manifest.Outcomes[2].State != "failed" || manifest.Outcomes[2].ErrorCode != "invalid_request" || len(fixture.runs) != 4 {
		t.Fatalf("permanent failure blocked members: %+v", manifest.Outcomes)
	}
	before := fixture.posts
	if err := invokeSweep(t, t.Context(), server.URL, path, "resume"); err == nil {
		t.Fatal("permanent member failure was hidden")
	}
	if fixture.posts != before {
		t.Fatal("permanent failure was automatically retried")
	}
}

func TestScanSweepRejectsChangedOrMalformedManifestBeforeDispatch(t *testing.T) {
	for _, change := range []string{"digest", "target", "unknown", "duplicate", "null", "oversized", "symlink"} {
		t.Run(change, func(t *testing.T) {
			fixture, server := newSweepFixture(t, 7)
			path := filepath.Join(t.TempDir(), "sweep.json")
			if err := invokeSweep(t, t.Context(), server.URL, path, "create", "--all"); err != nil {
				t.Fatal(err)
			}
			data, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			switch change {
			case "digest":
				data = bytes.Replace(data, []byte(`"snapshotDigest":"sha256:`), []byte(`"snapshotDigest":"sha257:`), 1)
			case "target":
				data = bytes.Replace(data, []byte("old.example"), []byte("new.example"), 1)
			case "unknown":
				data = append([]byte(`{"unknown":1,`), data[1:]...)
			case "duplicate":
				data = append([]byte(`{"schemaVersion":1,`), data[1:]...)
			case "null":
				data = []byte(`null`)
			case "oversized":
				data = bytes.Repeat([]byte("x"), maxSweepManifestBytes+1)
			case "symlink":
				target := path + ".original"
				if err := os.Rename(path, target); err != nil {
					t.Fatal(err)
				}
				if err := os.Symlink(target, path); err != nil {
					t.Fatal(err)
				}
			}
			if change != "symlink" {
				if err := os.WriteFile(path, data, 0600); err != nil {
					t.Fatal(err)
				}
			}
			before := fixture.posts
			if err := invokeSweep(t, t.Context(), server.URL, path, "resume"); err == nil {
				t.Fatal("invalid manifest accepted")
			}
			if fixture.posts != before {
				t.Fatal("invalid manifest dispatched")
			}
		})
	}
}

func TestScanSweepLockAndCreateOnlyPublication(t *testing.T) {
	path := filepath.Join(t.TempDir(), "sweep.json")
	unlock, err := lockSweepManifest(path)
	if err != nil {
		t.Fatal(err)
	}
	if other, err := lockSweepManifest(path); err == nil {
		other()
		t.Fatal("concurrent writer acquired manifest")
	}
	unlock()
	unlock, err = lockSweepManifest(path)
	if err != nil {
		t.Fatal(err)
	}
	unlock()
	fixture, server := newSweepFixture(t, 3)
	if err := invokeSweep(t, t.Context(), server.URL, path, "create", "--all"); err != nil {
		t.Fatal(err)
	}
	before := fixture.posts
	if err := invokeSweep(t, t.Context(), server.URL, path, "create", "--all"); err == nil {
		t.Fatal("manifest overwritten")
	}
	if fixture.posts != before {
		t.Fatal("existing snapshot replaced")
	}
}

func TestScanSweepRetryExhaustionRemainsResumable(t *testing.T) {
	fixture, server := newSweepFixture(t, 7)
	fixture.retryProject = "p0002"
	path := filepath.Join(t.TempDir(), "sweep.json")
	if err := invokeSweep(t, t.Context(), server.URL, path, "create", "--all"); err == nil {
		t.Fatal("exhausted retry hidden")
	}
	manifest, err := loadScanSweep(path)
	if err != nil {
		t.Fatal(err)
	}
	if manifest.Outcomes[2].State != "unknown" || fixture.posts != 7 || len(fixture.runs) != 4 {
		t.Fatalf("unbounded or incorrectly settled retries: %+v calls=%d", manifest.Outcomes, fixture.posts)
	}
	fixture.retryProject = ""
	fixture.changed = true
	if err := invokeSweep(t, t.Context(), server.URL, path, "resume"); err != nil {
		t.Fatal(err)
	}
	if fixture.posts != 8 || len(fixture.runs) != 5 {
		t.Fatal("unknown member did not resume independently")
	}
}

func TestScanSweepMissingInputSkipsOnlyItsProject(t *testing.T) {
	fixture, server := newSweepFixture(t, 7)
	fixture.missingArtifact = "p0002"
	path := filepath.Join(t.TempDir(), "sweep.json")
	if err := invokeSweep(t, t.Context(), server.URL, path, "create", "--all", "--artifact", "request=scan/input"); err != nil {
		t.Fatal(err)
	}
	manifest, err := loadScanSweep(path)
	if err != nil {
		t.Fatal(err)
	}
	if manifest.Outcomes[2].State != "skipped" || manifest.Outcomes[2].ErrorCode != "input_artifact_unavailable" || len(fixture.runs) != 4 {
		t.Fatalf("missing source blocked the scope: %+v", manifest.Outcomes)
	}
}

func TestScanSweepByteBoundsRefuseBeforeFirstPost(t *testing.T) {
	for _, scope := range []string{"request", "manifest"} {
		t.Run(scope, func(t *testing.T) {
			count, size := 7, maxSweepRequestBytes
			if scope == "manifest" {
				count, size = 1000, 8000
			}
			fixture, server := newSweepFixture(t, count)
			path := filepath.Join(t.TempDir(), "sweep.json")
			if err := invokeSweep(t, t.Context(), server.URL, path, "create", "--all", "--param", "tags="+strings.Repeat("x", size)); err == nil {
				t.Fatal("unbounded snapshot accepted")
			}
			if fixture.posts != 0 {
				t.Fatal("invalid snapshot dispatched before validation")
			}
			if _, err := os.Stat(path); !errors.Is(err, os.ErrNotExist) {
				t.Fatal("invalid snapshot persisted")
			}
		})
	}
}

func TestScanSweepInvalidReplacementPreservesPreviousManifest(t *testing.T) {
	_, server := newSweepFixture(t, 3)
	path := filepath.Join(t.TempDir(), "sweep.json")
	if err := invokeSweep(t, t.Context(), server.URL, path, "create", "--all"); err != nil {
		t.Fatal(err)
	}
	before, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	manifest, err := loadScanSweep(path)
	if err != nil {
		t.Fatal(err)
	}
	manifest.Snapshot.Members[0].Key = "changed"
	if err := saveScanSweep(path, &manifest, false); err == nil {
		t.Fatal("changed request accepted")
	}
	after, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(before, after) {
		t.Fatal("failed replacement lost previous snapshot")
	}
}
