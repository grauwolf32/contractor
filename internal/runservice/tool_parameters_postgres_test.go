package runservice

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/configload"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
)

func TestPostgresToolParameterProjectionValidatesBeforeCreationAndPreservesStrings(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedRunServicePool(t, ctx, databaseURL)
	catalogRoot := t.TempDir()
	for _, directory := range []string{"agent-templates", "workflows", "instructions", "model-policies", "llm-gateways", "execution-configs", "audit-profiles"} {
		if err := os.MkdirAll(filepath.Join(catalogRoot, directory), 0700); err != nil {
			t.Fatal(err)
		}
	}
	for _, relative := range []string{"workflows/nuclei.yaml", "agent-templates/nuclei.yaml", "instructions/scan.md"} {
		data, err := os.ReadFile(filepath.Join("..", "config", "testdata", "scan-catalog", relative))
		if err != nil {
			t.Fatal(err)
		}
		text := string(data)
		if strings.HasPrefix(relative, "workflows/") {
			text = strings.Replace(text, "target: {required: true}", "target: {required: true}\n    rate: {required: false}", 1)
		} else if strings.HasPrefix(relative, "agent-templates/") {
			text = strings.Replace(text, "rate_limit: {source: literal, value: 10}", "rate_limit: {source: parameter, name: rate, parse: integer, default: 7}", 1)
		}
		target := filepath.Join(catalogRoot, relative)
		if err := os.MkdirAll(filepath.Dir(target), 0700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(target, []byte(text), 0600); err != nil {
			t.Fatal(err)
		}
	}
	manager, _, err := configload.NewManager(config.ManagerOptions{OperatorRoot: catalogRoot, ManagedRoot: filepath.Join(t.TempDir(), "managed")})
	if err != nil {
		t.Fatal(err)
	}
	runs := runstore.NewPostgresStore(pool)
	service, err := New(Options{
		Runs: runs, Workflows: manager, LLMCredentials: emptyCredentialLookup{},
		CredentialGuard: openCredentialGuard{}, RuntimeCredentials: acceptRuntimeCredentials{},
		Projects: projectstore.NewPostgresStore(pool), PublicTransaction: postgresPublicTransaction(pool),
	})
	if err != nil {
		t.Fatal(err)
	}
	for index, rate := range []string{"7", "", "07", "+7", "7.0", "7 ", "9007199254740992", "omit"} {
		t.Run(fmt.Sprintf("case-%d", index), func(t *testing.T) {
			id := fmt.Sprintf("tool-parameter-run-%d", index)
			parameters := map[string]string{"target": "https://fixture.invalid/"}
			if rate != "omit" {
				parameters["rate"] = rate
			}
			params := PublicCreateParams{OwnerID: "tool-parameter-owner", Workflow: "nuclei-target@1", Parameters: parameters,
				Inputs: map[string]contracts.ArtifactRef{}, IdempotencyKey: id, NewRunID: func() (string, error) { return id, nil }}
			params.RequestDigest, err = PublicRequestDigest(params)
			if err != nil {
				t.Fatal(err)
			}
			result, err := service.CreatePublic(ctx, params)
			valid := rate == "7" || rate == "omit"
			if !valid {
				if !errors.Is(err, ErrInvalid) {
					t.Fatalf("invalid parameter error=%v", err)
				}
				if _, err := runs.GetRun(ctx, id); !errors.Is(err, runstore.ErrNotFound) {
					t.Fatalf("invalid input committed a Run: %v", err)
				}
				return
			}
			if err != nil || !result.Created {
				t.Fatalf("create=%+v error=%v", result, err)
			}
			stored, err := runs.GetRun(ctx, id)
			if err != nil {
				t.Fatal(err)
			}
			supplied, present := stored.Parameters["rate"]
			if rate == "omit" && present || rate == "7" && (!present || supplied != "7") {
				t.Fatal("stored strings were rewritten/defaulted")
			}
			replay, err := service.CreatePublic(ctx, params)
			if err != nil || !replay.Replayed || replay.Run.RunID != id {
				t.Fatalf("replay=%+v error=%v", replay, err)
			}
		})
	}
}
