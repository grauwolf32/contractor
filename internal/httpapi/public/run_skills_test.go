package public

import (
	"bytes"
	"encoding/json"
	"io/fs"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/agentskills"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/runstore"
)

func TestCreateRunCommitsExactSkillSelectionBeforeDeferredInitialization(t *testing.T) {
	root := copyPublicConfigTree(t)
	templatePath := filepath.Join(root, "agent-templates", "artifact_builder.yaml")
	template, err := os.ReadFile(templatePath)
	if err != nil {
		t.Fatal(err)
	}
	template = bytes.Replace(
		template,
		[]byte("  sandboxProfile: local-workdir@1\n"),
		[]byte("  skills: [{namespace: skills, name: review}]\n  sandboxProfile: local-workdir@1\n"),
		1,
	)
	if err := os.WriteFile(templatePath, template, 0o644); err != nil {
		t.Fatal(err)
	}
	fixture := newHandlerFixtureWithConfig(t, root)
	user, _ := fixture.artifacts.User("user-1")
	if _, err := user.Write(
		t.Context(), artifacts.ArtifactRef{Namespace: "projects", Name: "source"},
		artifacts.Payload{MediaType: "text/plain", Data: []byte("source")}, nil,
	); err != nil {
		t.Fatal(err)
	}
	selected, err := user.Write(
		t.Context(), artifacts.ArtifactRef{Namespace: contracts.AgentSkillNamespace, Name: "review"},
		artifacts.Payload{MediaType: agentskills.MediaType, Data: []byte("selected-package-canary")}, nil,
	)
	if err != nil {
		t.Fatal(err)
	}

	body := []byte(`{"workflow":"artifact-copy@1","parameters":{},"artifacts":{"source":{"namespace":"projects","name":"source"}},"executionConfig":{}}`)
	request := authenticatedRequest(http.MethodPost, "/v1/runs", bytes.NewReader(body))
	response := httptest.NewRecorder()
	fixture.handler.ServeHTTP(response, request)
	if response.Code != http.StatusAccepted || fixture.runSkills.calls != 1 || fixture.notifier.calls != 1 {
		t.Fatalf("create Skill Run = status %d, initializer %d, notifications %d, body %s", response.Code, fixture.runSkills.calls, fixture.notifier.calls, response.Body.String())
	}
	run := fixture.runs.runs["run_fixed"]
	if run.State != runstore.RunInitializing ||
		run.StateReason.Code != runstore.SkillInitializationPendingReason ||
		len(run.SkillSnapshot) != 1 || run.SkillSnapshot[0].Source == nil ||
		*run.SkillSnapshot[0].Source.Revision != *selected.Ref.Revision {
		t.Fatalf("durable Skill selection = %+v", run)
	}
	if strings.Contains(response.Body.String(), "selected-package-canary") {
		t.Fatal("Run response disclosed Skill bytes")
	}

	if _, err := user.Write(
		t.Context(), artifacts.ArtifactRef{Namespace: contracts.AgentSkillNamespace, Name: "review"},
		artifacts.Payload{MediaType: agentskills.MediaType, Data: []byte("new-package")},
		selected.Ref.Revision,
	); err != nil {
		t.Fatal(err)
	}
	replay := authenticatedRequest(http.MethodPost, "/v1/runs", bytes.NewReader(body))
	replayed := httptest.NewRecorder()
	fixture.handler.ServeHTTP(replayed, replay)
	if replayed.Code != http.StatusAccepted || replayed.Header().Get("Idempotency-Replayed") != "true" ||
		fixture.runSkills.calls != 1 {
		t.Fatalf("Skill Run replay = status %d, header %q, initializer %d, body %s", replayed.Code, replayed.Header().Get("Idempotency-Replayed"), fixture.runSkills.calls, replayed.Body.String())
	}
	if got := fixture.runs.runs["run_fixed"].SkillSnapshot[0].Source; got == nil ||
		*got.Revision != *selected.Ref.Revision {
		t.Fatalf("idempotency replay fell forward to another Skill revision: %+v", got)
	}
	var readModel createRunResponse
	if err := json.Unmarshal(replayed.Body.Bytes(), &readModel); err != nil ||
		readModel.State != runstore.RunInitializing {
		t.Fatalf("replay response = (%+v, %v)", readModel, err)
	}
}

func copyPublicConfigTree(t *testing.T) string {
	t.Helper()
	source := filepath.Clean("../../config/testdata/valid")
	target := filepath.Join(t.TempDir(), "configs")
	err := filepath.WalkDir(source, func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		relative, err := filepath.Rel(source, path)
		if err != nil {
			return err
		}
		destination := filepath.Join(target, relative)
		if entry.IsDir() {
			return os.MkdirAll(destination, 0o755)
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		return os.WriteFile(destination, data, 0o644)
	})
	if err != nil {
		t.Fatal(err)
	}
	return target
}
