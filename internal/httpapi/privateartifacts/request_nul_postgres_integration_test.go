//go:build integration

package privateartifacts

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/findingintake"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
)

func TestPostgresPrivateFindingNULEscapeStopsBeforeIntake(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	pool := isolatedPrivateArtifactPool(t, ctx)
	createPrivateArtifactRun(t, ctx, pool, "run-a")
	service, err := findingintake.New(pool, artifactpgtest.NewStorage())
	if err != nil {
		t.Fatal(err)
	}
	grant := testGrant("run-a")
	// FindReplay validates the complete submission, including its derived
	// identity, then reads the real receipt table. The positive fixture is
	// therefore accepted by production validation, not a permissive fake.
	if _, found, err := service.FindReplay(ctx, grant, privateNULSubmission(`a\u0000b`)); err != nil || found {
		t.Fatalf("valid literal-text submission replay = %t, %v", found, err)
	}
	intake := &privateNULIntakeCounter{FindingIntake: service}
	handler, err := newPayloadNativeRecorder(Dependencies{Transfers: artifactpgtest.NewStorage(),
		Registry:  &fakeRegistry{grant: grant},
		Artifacts: artifacts.NewService(artifactpgtest.NewStorage().Repository(pool)), Findings: intake,
	})
	if err != nil {
		t.Fatal(err)
	}
	body, err := json.Marshal(privateNULSubmission("a\x00b"))
	if err != nil {
		t.Fatal(err)
	}
	request := trustedRequest(http.MethodPost, "/private/v1/allocations/allocation-1/finding-proposals", bytes.NewReader(body))
	request = request.WithContext(ctx)
	request.Header.Set("Content-Type", "application/json")
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	assertPrivateErrorEnvelope(t, response, http.StatusBadRequest, "invalid_finding_request", false)
	if intake.replays != 0 || intake.submits != 0 {
		t.Fatalf("NUL reached real intake: replays=%d submits=%d", intake.replays, intake.submits)
	}
	var receipts, bindings int
	if err := pool.QueryRow(ctx, `SELECT
  (SELECT count(*) FROM finding_proposal_receipts),
  (SELECT count(*) FROM artifact_bindings)`).Scan(&receipts, &bindings); err != nil {
		t.Fatal(err)
	}
	if receipts != 0 || bindings != 0 {
		t.Fatalf("NUL request mutated intake: receipts=%d bindings=%d", receipts, bindings)
	}
}

type privateNULIntakeCounter struct {
	FindingIntake
	replays, submits int
}

func (c *privateNULIntakeCounter) FindReplay(ctx context.Context, grant controlplane.AllocationGrant, input findingintake.Submission) (findingintake.Receipt, bool, error) {
	c.replays++
	return c.FindingIntake.FindReplay(ctx, grant, input)
}

func (c *privateNULIntakeCounter) Submit(ctx context.Context, grant controlplane.AllocationGrant, input findingintake.Submission) (findingintake.Receipt, bool, error) {
	c.submits++
	return c.FindingIntake.Submit(ctx, grant, input)
}

func TestPostgresPrivateJSONNULErrorIsPermanent(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	pool := isolatedPrivateArtifactPool(t, ctx)
	var decoded []byte
	err := pool.QueryRow(ctx, `SELECT $1::jsonb`, `{"description":"a\u0000b"}`).Scan(&decoded)
	if err == nil || persistencepostgres.Classify(err) != persistencepostgres.ClassInvalid {
		t.Fatalf("PostgreSQL NUL jsonb cast = %v, want permanent data exception", err)
	}
	response := httptest.NewRecorder()
	(&handler{}).handleError(response, fmt.Errorf("write JSON metadata: %w", err))
	assertPrivateErrorEnvelope(t, response, http.StatusBadRequest, "invalid_request", false)
}

func TestPostgresPrivateArtifactBytesKeepNULAndJSONEscapes(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	pool := isolatedPrivateArtifactPool(t, ctx)
	createPrivateArtifactRun(t, ctx, pool, "run-a")
	handler, err := newPayloadNativeRecorder(Dependencies{Transfers: artifactpgtest.NewStorage(),
		Registry:  &fakeRegistry{grant: testGrant("run-a")},
		Artifacts: artifacts.NewService(artifactpgtest.NewStorage().Repository(pool)),
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, test := range []struct {
		name string
		data []byte
	}{
		{"raw-nul", []byte("before\x00after")},
		{"json-nul", []byte(`{"note":"\u0000"}`)},
		{"literal-backslash", []byte(`{"note":"\\u0000"}`)},
	} {
		t.Run(test.name, func(t *testing.T) {
			written := putArtifact(t, handler, "builder", test.name, "*", test.data)
			if written.Code != http.StatusCreated {
				t.Fatalf("artifact write = %d %s", written.Code, written.Body.String())
			}
			response := httptest.NewRecorder()
			request := trustedRequest(http.MethodGet, "/private/v1/allocations/allocation-1/artifacts/builder/"+test.name, nil)
			handler.ServeHTTP(response, request.WithContext(ctx))
			if response.Code != http.StatusOK || !bytes.Equal(response.Body.Bytes(), test.data) {
				t.Fatalf("artifact bytes = %d %q, want %q", response.Code, response.Body.Bytes(), test.data)
			}
		})
	}
}
