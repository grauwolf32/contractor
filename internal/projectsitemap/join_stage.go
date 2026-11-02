package projectsitemap

import (
	"context"
	"encoding/json"
	"errors"

	"github.com/grauwolf32/contractor/internal/artifactpolicy"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	postgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/scanplan"
	"github.com/jackc/pgx/v5"
)

// StageJoin names only current-Run exact inputs and declared ordinary outputs.
// All authority comes from the owned Run/Stage, not an allocation or model.
type StageJoin struct {
	RunID   string                 `json:"runId"`
	StageID string                 `json:"stageId"`
	ClaimID string                 `json:"-"`
	Sitemap contracts.ArtifactRef  `json:"sitemap"`
	OpenAPI contracts.ArtifactRef  `json:"openapi"`
	Output  *contracts.ArtifactRef `json:"output,omitempty"`
	Report  *contracts.ArtifactRef `json:"report,omitempty"`
	Options JoinOptions            `json:"options"`
}

type StageJoinResult struct {
	Output      *contracts.ArtifactRef `json:"output,omitempty"`
	Report      *contracts.ArtifactRef `json:"report,omitempty"`
	Summary     JoinSummary            `json:"summary"`
	Replayed    bool                   `json:"replayed"`
	Publication *contracts.ArtifactRef `json:"publication,omitempty"`
}

// JoinStage computes an immutable result without changing the live Project map.
func (s *Service) JoinStage(ctx context.Context, request StageJoin) (StageJoinResult, error) {
	return s.joinStage(ctx, request, false)
}

// ApplyContractStage explicitly updates only contract declarations/grouping on
// the current map. Expected source state remains fixed by the selected input.
func (s *Service) ApplyContractStage(ctx context.Context, request StageJoin) (StageJoinResult, error) {
	return s.joinStage(ctx, request, true)
}

func (s *Service) joinStage(ctx context.Context, request StageJoin, apply bool) (StageJoinResult, error) {
	if request.Sitemap.ValidateExact() != nil || request.OpenAPI.ValidateExact() != nil || !contracts.ValidResourceID(request.RunID) || !contracts.ValidResourceID(request.StageID) {
		return StageJoinResult{}, ErrInvalid
	}
	if request.Output != nil && (request.Output.Validate() != nil || request.Output.Revision != nil || artifactpolicy.IsPurposeReservedNamespace(request.Output.Namespace)) {
		return StageJoinResult{}, ErrInvalid
	}
	if request.Report != nil && (request.Report.Validate() != nil || request.Report.Revision != nil || artifactpolicy.IsPurposeReservedNamespace(request.Report.Namespace)) {
		return StageJoinResult{}, ErrInvalid
	}
	digest, err := contentdigest.JSON(struct {
		Request StageJoin
		Apply   bool
	}{request, apply})
	if err != nil {
		return StageJoinResult{}, err
	}
	var result StageJoinResult
	err = s.mutationTransaction(ctx, apply, func(ctx context.Context, tx pgx.Tx) error {
		store := runstore.NewPostgresTxStore(tx)
		owner, err := store.ProjectContext(ctx, request.RunID)
		if err != nil {
			return err
		}
		if err := checkProject(ctx, tx, owner.OwnerID, owner.ProjectID, true); err != nil {
			return err
		}
		execution, err := store.LockStageOperation(ctx, request.RunID, request.StageID, request.ClaimID)
		if err != nil {
			return err
		}
		if execution.State != runstore.StageRunning {
			return runstore.ErrConflict
		}
		for name, ref := range map[string]contracts.ArtifactRef{"sitemap": request.Sitemap, "openapi": request.OpenAPI} {
			input, ok := execution.StageContext.Artifacts[name]
			if !ok || input.Artifact == nil || !input.Artifact.SameExact(ref) {
				return ErrInvalid
			}
		}
		key := postgres.ProjectReceiptKey(owner.OwnerID, owner.ProjectID, owner.ProjectID, "sitemap.join", request.StageID)
		receipt, replayed, err := postgres.KeyedMutation(ctx, tx, key, digest, func() (postgres.Receipt, error) {
			admitted, release, err := artifacts.AcquireTransfer(ctx)
			if err != nil {
				return postgres.Receipt{}, err
			}
			defer release()
			ctx = admitted
			service := artifacts.NewService(artifacts.NewPostgresTxRepository(tx))
			consumer, err := service.Run(request.RunID)
			if err != nil {
				return postgres.Receipt{}, err
			}
			metadata, err := consumer.Metadata(ctx, request.OpenAPI)
			if err != nil {
				return postgres.Receipt{}, err
			}
			if metadata.Size > scanplan.MaxSourceBytes {
				return postgres.Receipt{}, ErrInvalid
			}
			mapMetadata, err := consumer.Metadata(ctx, request.Sitemap)
			if err != nil {
				return postgres.Receipt{}, err
			}
			if mapMetadata.Size > MaxDocumentBytes || mapMetadata.MediaType != MediaType {
				return postgres.Receipt{}, ErrInvalid
			}
			input, err := consumer.ReadExact(ctx, metadata.Exact())
			if err != nil {
				return postgres.Receipt{}, err
			}
			selected, err := consumer.ReadExact(ctx, mapMetadata.Exact())
			if err != nil {
				return postgres.Receipt{}, err
			}
			document, err := Decode(selected.Payload.Data)
			if err != nil {
				return postgres.Receipt{}, err
			}
			computed, err := joinOpenAPI(ctx, document, input.Payload.Data, input.Payload.MediaType, metadata.Exact(), request.Options)
			if err != nil {
				return postgres.Receipt{}, err
			}
			accepted := StageJoinResult{Summary: computed.summary}
			if apply {
				project, err := service.Project(owner.ProjectID)
				if err != nil {
					return postgres.Receipt{}, err
				}
				previous, current, err := readMap(ctx, project)
				if err != nil {
					return postgres.Receipt{}, err
				}
				if manualIntent(current) != manualIntent(document) {
					return postgres.Receipt{}, ErrConflict
				}
				authority := computed.authority
				authority.OwnerID = owner.OwnerID
				authority.ProjectID = owner.ProjectID
				authority.RunID = request.RunID
				_, payload, err := Reduce(current, computed.mutation, authority)
				if err != nil {
					return postgres.Receipt{}, err
				}
				published, err := publishMap(ctx, service, owner.ProjectID, previous, payload)
				if err != nil {
					return postgres.Receipt{}, err
				}
				accepted.Publication = &published.Ref
				computed.payload = payload
			}
			scope, _ := artifacts.RunScope(request.RunID)
			if request.Output != nil {
				written, err := service.Create(ctx, scope, *request.Output, artifacts.Payload{MediaType: MediaType, Data: computed.payload})
				if errors.Is(err, artifacts.ErrArtifactConflict) {
					return postgres.Receipt{}, ErrConflict
				}
				if err != nil {
					return postgres.Receipt{}, err
				}
				accepted.Output = &written.Ref
			}
			if request.Report != nil {
				data, err := json.Marshal(struct {
					Schema      string                  `json:"schema"`
					Sitemap     contracts.ExactArtifact `json:"sitemap"`
					OpenAPI     contracts.ExactArtifact `json:"openapi"`
					Summary     JoinSummary             `json:"summary"`
					Publication *contracts.ArtifactRef  `json:"publication,omitempty"`
				}{"sitemap-openapi-comparison@1", mapMetadata.Exact(), metadata.Exact(), computed.summary, accepted.Publication})
				if err != nil || len(data) > MaxBatchBytes {
					return postgres.Receipt{}, ErrInvalid
				}
				report, err := service.Create(ctx, scope, *request.Report, artifacts.Payload{MediaType: "application/json", Data: data})
				if errors.Is(err, artifacts.ErrArtifactConflict) {
					return postgres.Receipt{}, ErrConflict
				}
				if err != nil {
					return postgres.Receipt{}, err
				}
				accepted.Report = &report.Ref
			}
			// The Run lock prevents another owner taking over. Recheck the lease
			// at publication completion so an expired claim cannot commit effects.
			if _, err := store.LockStageOperation(ctx, request.RunID, request.StageID, request.ClaimID); err != nil {
				return postgres.Receipt{}, err
			}
			response, err := json.Marshal(accepted)
			return postgres.Receipt{OwnerID: owner.OwnerID, ProjectID: owner.ProjectID, ResourceID: owner.ProjectID, Operation: "sitemap.join", Response: response}, err
		})
		if err != nil {
			return err
		}
		if err := strictjson.Decode(receipt.Response, &result); err != nil {
			return ErrInvalid
		}
		result.Replayed = replayed
		return nil
	})
	return result, err
}

// manualIntent excludes observations and grouping, which may advance while a
// selected contract is computed. Edits and explicit suppression remain part of
// the selected intent even when their containing groups move.
func manualIntent(document Document) string {
	type selected struct {
		Sources      []Source
		Rules        []Rule
		Markers      []Marker
		Annotations  map[string][]Annotation
		Declarations map[string][]Declaration
	}
	value := selected{Markers: document.Markers, Annotations: map[string][]Annotation{}, Declarations: map[string][]Declaration{}}
	manual := map[string]bool{}
	for _, source := range document.Sources {
		if source.Provenance.Kind == "manual" {
			value.Sources = append(value.Sources, source)
			manual[source.Provenance.SourceID] = true
		}
	}
	for _, rule := range document.Rules {
		if manual[rule.SourceID] {
			value.Rules = append(value.Rules, rule)
		}
	}
	for _, group := range document.Groups {
		for _, declaration := range group.Declarations {
			if declaration.Provenance.Kind == "manual" {
				value.Declarations[group.ID] = append(value.Declarations[group.ID], declaration)
			}
		}
		for _, member := range group.Members {
			if len(member.Annotations) > 0 {
				value.Annotations[member.ID] = member.Annotations
			}
		}
	}
	return contentdigest.Bytes(jsonBytes(value))
}
