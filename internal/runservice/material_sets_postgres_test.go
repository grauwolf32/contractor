package runservice

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/configload"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
)

func TestPostgresMaterialSetsUseOrdinaryRunCreationAndAtomicRejection(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(t.Context(), 2*time.Minute)
	defer cancel()
	pool := isolatedRunServicePool(t, ctx, databaseURL)
	root := t.TempDir()
	for _, dir := range []string{"workflows", "agent-templates", "instructions", "model-policies", "llm-gateways", "audit-profiles"} {
		if e := os.MkdirAll(filepath.Join(root, dir), 0700); e != nil {
			t.Fatal(e)
		}
	}
	for _, file := range []string{"workflows/nuclei.yaml", "agent-templates/nuclei.yaml", "instructions/scan.md"} {
		raw, e := os.ReadFile(filepath.Join("..", "config", "testdata", "scan-catalog", file))
		if e != nil {
			t.Fatal(e)
		}
		text := string(raw)
		if strings.HasPrefix(file, "workflows/") {
			text = strings.Replace(text, "inputs: {}", "inputs:\n    materials: {required: true, materialSet: true, mediaTypes: [text/plain]}\n    second: {required: false, materialSet: true, mediaTypes: [text/plain]}\n    singleton: {required: false, mediaTypes: [text/plain]}", 1)
		}
		if e := os.WriteFile(filepath.Join(root, file), []byte(text), 0600); e != nil {
			t.Fatal(e)
		}
	}
	manager, _, e := configload.NewManager(config.ManagerOptions{OperatorRoot: root, ManagedRoot: filepath.Join(t.TempDir(), "managed")})
	if e != nil {
		t.Fatal(e)
	}
	runs := runstore.NewPostgresStore(pool, artifactpgtest.NewStorage())
	store := artifacts.NewService(artifactpgtest.NewStorage().Repository(pool))
	user, _ := store.User("material-owner")
	members := make([]contracts.MaterialDescriptor, 33)
	for i := range members {
		m := writeExact(t, ctx, user, "reports", fmt.Sprintf("report-%02d", i), "text/plain", []byte(fmt.Sprint("exact source ", i)))
		members[i] = contracts.MaterialDescriptor{ScopeKind: "user", ScopeID: "material-owner", Ref: m.Ref, Digest: m.Digest, MediaType: m.MediaType, SizeBytes: m.SizeBytes}
	}
	lostResponse := false
	transaction := postgresPublicTransaction(pool)
	options := Options{Runs: runs, Workflows: manager, LLMCredentials: emptyCredentialLookup{}, CredentialGuard: openCredentialGuard{}, RuntimeCredentials: acceptRuntimeCredentials{}, Projects: projectstore.NewPostgresStore(pool), PublicTransaction: func(ctx context.Context, fn func(PublicRunWriter, *artifacts.Service) error) error {
		err := transaction(ctx, fn)
		if err == nil && lostResponse {
			lostResponse = false
			return io.ErrUnexpectedEOF
		}
		return err
	}}
	service, e := New(options)
	if e != nil {
		t.Fatal(e)
	}
	for _, scenario := range []struct {
		name    string
		count   int
		invalid bool
	}{{"one", 1, false}, {"seventeen", 17, false}, {"ceiling", 32, false}, {"overflow", 33, true}, {"required-empty", 0, true}, {"duplicate", 2, false}, {"foreign", 2, true}, {"hidden", 2, true}, {"media", 2, true}, {"metadata", 2, true}, {"missing", 2, true}, {"versionless", 1, true}, {"lost-response", 17, false}, {"pair-ceiling", 17, false}, {"pair-overflow", 17, true}, {"pair-bytes", 17, true}, {"singleton-overflow", 32, true}, {"singleton-duplicate", 32, false}} {
		t.Run(scenario.name, func(t *testing.T) {
			selected := append([]contracts.MaterialDescriptor{}, members[:scenario.count]...)
			switch scenario.name {
			case "duplicate":
				selected[1] = selected[0]
			case "foreign":
				selected[1].ScopeID = "foreign-owner"
			case "hidden":
				selected[1].Ref.Namespace = "skills"
			case "media":
				selected[1].MediaType = "application/zip"
			case "metadata":
				selected[1].Digest = testDigest("0")
			case "missing":
				selected[1].Ref.Name = "missing"
			}
			raw, e := json.Marshal(contracts.MaterialSet{Schema: contracts.MaterialSetSchema, Materials: selected})
			if e != nil {
				t.Fatal(e)
			}
			if scenario.name == "pair-bytes" {
				raw = append(raw, []byte(strings.Repeat(" ", 600000))...)
			}
			source := writeExact(t, ctx, user, "selections", scenario.name, "application/json", raw)
			params := PublicCreateParams{OwnerID: "material-owner", Workflow: "nuclei-target@1", Parameters: map[string]string{"target": "https://fixture.invalid/"}, Inputs: map[string]contracts.ArtifactRef{"materials": source.Ref}, IdempotencyKey: "set-" + scenario.name, NewRunID: func() (string, error) { return "run-set-" + scenario.name, nil }}
			if scenario.name == "pair-ceiling" || scenario.name == "pair-overflow" || scenario.name == "pair-bytes" {
				stop := 32
				if scenario.name == "pair-overflow" {
					stop = 33
				}
				secondRaw, _ := json.Marshal(contracts.MaterialSet{Schema: contracts.MaterialSetSchema, Materials: members[16:stop]})
				if scenario.name == "pair-bytes" {
					secondRaw = append(secondRaw, []byte(strings.Repeat(" ", 600000))...)
				}
				second := writeExact(t, ctx, user, "selections", scenario.name+"-second", "application/json", secondRaw)
				params.Inputs["second"] = second.Ref
			}
			if scenario.name == "singleton-overflow" {
				params.Inputs["singleton"] = members[32].Ref
			}
			if scenario.name == "singleton-duplicate" {
				params.Inputs["singleton"] = members[0].Ref
			}
			if scenario.name == "versionless" {
				source.Ref.Revision = nil
				params.Inputs["materials"] = source.Ref
			}
			params.RequestDigest, e = PublicRequestDigest(params)
			if e != nil {
				t.Fatal(e)
			}
			if scenario.name == "lost-response" {
				lostResponse = true
			}
			made, e := service.CreatePublic(ctx, params)
			if scenario.name == "lost-response" {
				if !errors.Is(e, io.ErrUnexpectedEOF) {
					t.Fatal("missing acknowledgement fault", e)
				}
				// A fresh service instance must reconcile the original receipt
				// before touching mutable context or choosing another Run ID.
				service, e = New(options)
				if e != nil {
					t.Fatal(e)
				}
				made, e = service.CreatePublic(ctx, params)
				if e != nil || !made.Replayed || made.Run.RunID != "run-set-lost-response" {
					t.Fatal(made, e)
				}
			}

			if scenario.invalid {
				if e == nil {
					t.Fatal("invalid set dispatched")
				}
				var n int
				if e := pool.QueryRow(ctx, "SELECT count(*) FROM workflow_runs WHERE run_id=$1", "run-set-"+scenario.name).Scan(&n); e != nil || n != 0 {
					t.Fatalf("partial Run=%d %v", n, e)
				}
				return
			}
			if e != nil || !made.Created && scenario.name != "lost-response" {
				t.Fatal(made, e)
			}
			run, _ := store.Run(made.Run.RunID)
			view, e := run.Read(ctx, contracts.ArtifactRef{Namespace: "inputs", Name: "materials"})
			if e != nil {
				t.Fatal(e)
			}
			local, e := contracts.DecodeMaterialSet(view.Payload.Data)
			if e != nil {
				t.Fatal(e)
			}
			want := scenario.count
			if scenario.name == "duplicate" {
				want = 1
			}
			if len(local.Materials) != want {
				t.Fatal("set size", len(local.Materials), want)
			}
			for _, m := range local.Materials {
				if m.ScopeKind != "run" || m.ScopeID != made.Run.RunID || m.Ref.Namespace != "material-inputs" {
					t.Fatal("not a readable local member", m)
				}
				read, e := run.Read(ctx, m.Ref)
				if e != nil || m.Exact().Verify(read.Exact()) != nil {
					t.Fatal("unreadable exact member", e)
				}
			}
			lineage, e := run.DescribeForkSources(ctx, "inputs", []string{"materials"}, artifacts.LineageInputFork)
			if e != nil || len(lineage) != 1 || !lineage[0].Source.SameExact(source.Ref) {
				t.Fatal("repeat lost original selection", lineage, e)
			}
			replay, e := service.CreatePublic(ctx, params)
			if e != nil || !replay.Replayed || replay.Run.RunID != made.Run.RunID {
				t.Fatal("response loss changed identity", replay, e)
			}
			var versions int
			if e := pool.QueryRow(ctx, "SELECT count(*) FROM artifact_versions WHERE media_type='text/plain'").Scan(&versions); e != nil || versions != 33 {
				t.Fatalf("material fork duplicated content versions: versions=%d err=%v", versions, e)
			}
		})
	}
}
