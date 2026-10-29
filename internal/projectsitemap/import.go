package projectsitemap

import (
	"context"
	"encoding/json"
	"slices"
	"strings"
	"time"

	"github.com/grauwolf32/contractor/internal/artifactpolicy"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contentdigest"
	"github.com/grauwolf32/contractor/internal/contracts"
	postgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/strictjson"
	"github.com/jackc/pgx/v5"
)

const (
	maxScannerReportBytes = 512 << 10
	maxScannerResultBytes = 128 << 10
)

// ImportSelection names existing normalized tool output. Scope, source kind and
// deployment come from the allocation and the closed format, never model JSON.
type ImportSelection struct {
	Format   string                `json:"format" strictjson:"required"`
	Artifact contracts.ArtifactRef `json:"artifact" strictjson:"required"`
}

func (s ImportSelection) Validate() error {
	if !slices.Contains([]string{"ffuf-report@1", "katana-report@1"}, s.Format) || s.Artifact.ValidateExact() != nil {
		return ErrInvalid
	}
	return nil
}

// ImportAllocation never starts a scanner or reads another Run. The keyed
// transaction replays before source/map reads, including after history eviction.
func (s *Service) ImportAllocation(ctx context.Context, access AllocationAccess, idempotencyKey string, selection ImportSelection) (Result, error) {
	if selection.Validate() != nil || access.Policy.Validate() != nil || !contracts.ValidIdempotencyKey(idempotencyKey) {
		return Result{}, ErrInvalid
	}
	if artifactpolicy.IsRunSystemNamespace(selection.Artifact.Namespace) {
		return Result{}, artifacts.ErrReservedNamespace
	}
	kind := strings.TrimSuffix(selection.Format, "-report@1")
	authority := observationAuthority(access, kind)
	if authority.validate() != nil {
		return Result{}, ErrInvalid
	}
	digest, err := contentdigest.JSON(struct {
		Selection   ImportSelection
		Provenance  Provenance
		ProducerRun string
	}{selection, authority.provenance(), access.RunID})
	if err != nil {
		return Result{}, err
	}
	return s.mutationTransaction(ctx, true, func(ctx context.Context, tx pgx.Tx) (Result, error) {
		if err := checkProject(ctx, tx, access.OwnerID, access.ProjectID, true); err != nil {
			return Result{}, err
		}
		receiptKey := postgres.ProjectReceiptKey(access.OwnerID, access.ProjectID, access.ProjectID, "sitemap.import", idempotencyKey)
		receipt, replayed, err := postgres.KeyedMutation(ctx, tx, receiptKey, digest, func() (postgres.Receipt, error) {
			admitted, release, err := artifacts.AcquireTransfer(ctx)
			if err != nil {
				return postgres.Receipt{}, err
			}
			defer release()
			service := artifacts.NewService(artifacts.NewPostgresTxRepository(tx))
			run, err := service.Run(access.RunID)
			if err != nil {
				return postgres.Receipt{}, err
			}
			metadata, err := run.Metadata(admitted, selection.Artifact)
			if err != nil {
				return postgres.Receipt{}, err
			}
			if metadata.Size <= 0 || metadata.Size > maxScannerReportBytes || metadata.MediaType != "application/json" {
				return postgres.Receipt{}, ErrInvalid
			}
			report, err := run.ReadExact(admitted, metadata.Exact())
			if err != nil {
				return postgres.Receipt{}, err
			}
			command, err := scannerReport(report.Payload.Data, kind, report.RevisionCreatedAt.UTC().Format(time.RFC3339Nano), authority.SourceID)
			if err != nil {
				return postgres.Receipt{}, err
			}
			project, err := service.Project(access.ProjectID)
			if err != nil {
				return postgres.Receipt{}, err
			}
			previous, current, err := readMap(admitted, project)
			if err != nil {
				return postgres.Receipt{}, err
			}
			next := current
			if len(command.Observations) != 0 || len(command.Gaps) != 0 {
				next, _, err = Reduce(current, command, authority)
				if err != nil {
					return postgres.Receipt{}, err
				}
				attachReportEvidence(&next, command.Observations, authority.SourceID, Evidence{RunID: access.RunID, Artifact: report.Exact()})
			}
			data, err := encode(next)
			if err != nil {
				return postgres.Receipt{}, err
			}
			result, err := publishMap(admitted, service, access.ProjectID, previous, data)
			if err != nil {
				return postgres.Receipt{}, err
			}
			response, err := json.Marshal(receiptRecord{Result: result, Policy: []Rule{}})
			return postgres.Receipt{OwnerID: access.OwnerID, ProjectID: access.ProjectID, ResourceID: access.ProjectID, Operation: "sitemap.import", Response: response}, err
		})
		if err != nil {
			return Result{}, err
		}
		var record receiptRecord
		if err := strictjson.Decode(receipt.Response, &record); err != nil {
			return Result{}, ErrInvalid
		}
		record.Result.Replayed = replayed
		return record.Result, nil
	})
}

func attachReportEvidence(document *Document, observations []Observation, sourceID string, evidence Evidence) {
	selected := map[string]bool{}
	for _, observation := range observations {
		origin, path, _, err := projectURL(observation.URL)
		if err != nil {
			continue // The strict importer validated this URL before reduction.
		}
		route, _ := routeFor(document.Rules, origin, observation.Method, path)
		structural, err := projectStructural(document.Rules, origin, observation.Method, route, observation.URL)
		if err == nil {
			selected[memberID(origin, observation.Method, path, structural...)] = true
		}
	}
	for i := range document.Groups {
		for j := range document.Groups[i].Members {
			member := &document.Groups[i].Members[j]
			if !selected[member.ID] || slices.ContainsFunc(member.Evidence, func(existing Evidence) bool {
				return existing.RunID == evidence.RunID && existing.Artifact.Ref.SameExact(evidence.Artifact.Ref)
			}) {
				continue
			}
			if len(member.Evidence) == MaxProvenance {
				addGap(document, sourceID, "evidence", 1)
				continue
			}
			member.Evidence = append(member.Evidence, evidence)
		}
	}
}
