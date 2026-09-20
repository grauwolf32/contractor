package scan

import (
	"context"
	"errors"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts"
)

type adapterRepository struct {
	artifacts.Repository
	artifacts.QueryRepository
	t             *testing.T
	metadata      artifacts.Metadata
	read          artifacts.ReadResult
	write         artifacts.WriteResult
	writeErr      error
	reads, writes int
}

func (r *adapterRepository) scope(scope artifacts.Scope) {
	r.t.Helper()
	if scope.Kind() != artifacts.ScopeRun || scope.ID() != "run-adapter" {
		r.t.Fatalf("incorrect artifact scope: %v", scope)
	}
}
func (r *adapterRepository) Metadata(_ context.Context, scope artifacts.Scope, _ contracts.ArtifactRef) (artifacts.Metadata, error) {
	r.scope(scope)
	return r.metadata, nil
}
func (r *adapterRepository) Read(_ context.Context, scope artifacts.Scope, ref contracts.ArtifactRef) (artifacts.ReadResult, error) {
	r.scope(scope)
	r.reads++
	if !sameRef(ref, r.metadata.Ref) {
		r.t.Fatal("read did not pin metadata revision")
	}
	return r.read, nil
}
func (r *adapterRepository) Write(_ context.Context, scope artifacts.Scope, ref contracts.ArtifactRef, _ artifacts.Payload, expected *string) (artifacts.WriteResult, error) {
	r.scope(scope)
	r.writes++
	if expected != nil || ref.Revision != nil {
		r.t.Fatal("planner attempted to overwrite an artifact")
	}
	return r.write, r.writeErr
}

func adapterFixture(t *testing.T) (*ServiceArtifacts, *adapterRepository, contracts.ArtifactRef) {
	t.Helper()
	revision := "revision-1"
	ref := contracts.ArtifactRef{Namespace: "planner", Name: "report", Revision: &revision}
	repository := &adapterRepository{t: t,
		metadata: artifacts.Metadata{Ref: ref, MediaType: "application/json", Size: 2},
		read:     artifacts.ReadResult{Ref: ref, Payload: artifacts.Payload{MediaType: "application/json", Data: []byte("{}")}},
		write:    artifacts.WriteResult{Ref: ref, MediaType: "application/json", Size: 2},
	}
	adapter, err := NewServiceArtifacts(artifacts.NewService(repository))
	if err != nil {
		t.Fatal(err)
	}
	return adapter, repository, ref
}

func TestArtifactReadBoundsAndExactRevision(t *testing.T) {
	for _, name := range []string{"valid", "oversized", "wrong revision", "wrong media", "wrong size", "unversioned"} {
		t.Run(name, func(t *testing.T) {
			adapter, repository, ref := adapterFixture(t)
			switch name {
			case "oversized":
				repository.metadata.Size = 1025
			case "wrong revision":
				revision := "revision-2"
				repository.read.Ref.Revision = &revision
			case "wrong media":
				repository.read.Payload.MediaType = "text/plain"
			case "wrong size":
				repository.read.Payload.Data = []byte("[] ")
			case "unversioned":
				ref.Revision = nil
			}
			payload, err := adapter.Read(t.Context(), "run-adapter", ref, 1024)
			if name == "valid" {
				if err != nil || string(payload.Data) != "{}" {
					t.Fatalf("valid read: %v", err)
				}
				payload.Data[0] = '['
				if string(repository.read.Payload.Data) != "{}" {
					t.Fatal("read aliased repository bytes")
				}
			} else if err == nil {
				t.Fatal("invalid read accepted")
			}
			if (name == "oversized" || name == "unversioned") && repository.reads != 0 {
				t.Fatal("invalid metadata caused a payload read")
			}
		})
	}
}

func TestArtifactCreateOnlyRecoveryComparesExactContent(t *testing.T) {
	for _, name := range []string{"created", "same content", "changed content", "changed media", "ambiguous write"} {
		t.Run(name, func(t *testing.T) {
			adapter, repository, ref := adapterFixture(t)
			if name != "created" {
				repository.writeErr = artifacts.ErrArtifactConflict
			}
			switch name {
			case "changed content":
				repository.read.Payload.Data = []byte("[]")
			case "changed media":
				repository.metadata.MediaType = "text/plain"
			case "ambiguous write":
				repository.writeErr = errors.New("connection lost")
			}
			target := ref
			target.Revision = nil
			got, err := adapter.Create(t.Context(), "run-adapter", target, artifacts.Payload{MediaType: "application/json", Data: []byte("{}")})
			if name == "created" || name == "same content" {
				if err != nil || !sameRef(got, ref) {
					t.Fatalf("creation/recovery = %v, %v", got, err)
				}
			} else if err == nil {
				t.Fatal("conflicting or uncertain creation accepted")
			}
			if repository.writes != 1 {
				t.Fatal("creation retried an ambiguous write")
			}
		})
	}
}

func TestFailedReportResolutionPinsOnlyItsOwnBinding(t *testing.T) {
	adapter, repository, ref := adapterFixture(t)
	target := ref
	target.Revision = nil
	got, err := adapter.Resolve(t.Context(), "run-adapter", target)
	if err != nil || !sameRef(got, ref) || repository.reads != 0 {
		t.Fatalf("resolve = %v, %v", got, err)
	}
	repository.metadata.Ref.Name = "other-job"
	if _, err := adapter.Resolve(t.Context(), "run-adapter", target); err == nil {
		t.Fatal("resolved another job report")
	}
}

func TestObservationClassificationIsDeterministic(t *testing.T) {
	var report workerReport
	if !strictWorkerReport([]byte(`{"schemaVersion":1,"tool":"scan.nuclei","inputDigest":"unused","inputArtifacts":{},"observation":{"status":"completed","exitCode":0,"stdoutTruncated":true,"resultsTruncated":null}}`), &report) {
		t.Fatal("fixture failed to decode")
	}
	for range 100 {
		status, code := observationStatus(report.Observation)
		if status != "incomplete" || code != "scan_report_invalid" {
			t.Fatalf("classification = %s, %s", status, code)
		}
	}
}

func TestWorkerReportRejectsCaseAliasesAndNullFields(t *testing.T) {
	for _, data := range []string{
		`{"schemaVersion":1,"tool":"scan_nuclei","Tool":"scan_sqlmap","inputDigest":"digest","inputArtifacts":{},"observation":{}}`,
		`{"schemaVersion":1,"Tool":"scan_nuclei","inputDigest":"digest","inputArtifacts":{},"observation":{}}`,
		`{"schemaVersion":1,"tool":"scan_nuclei","inputDigest":"digest","inputArtifacts":null,"observation":{}}`,
	} {
		var report workerReport
		if strictWorkerReport([]byte(data), &report) {
			t.Fatal("accepted a noncanonical report shape")
		}
	}
}
