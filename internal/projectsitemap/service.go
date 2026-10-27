package projectsitemap

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"slices"
	"strings"

	"github.com/grauwolf32/contractor/internal/artifactpolicy"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contentdigest"
	"github.com/grauwolf32/contractor/internal/contracts"
	postgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/strictjson"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// Service is the production publication boundary. Every store involved in a
// mutation uses the same transaction, including the common Artifact writer.
type Service struct{ pool *pgxpool.Pool }

func NewService(pool *pgxpool.Pool) *Service { return &Service{pool: pool} }

type View struct {
	Revision string   `json:"revision"`
	Document Document `json:"document"`
}
type Result struct {
	Ref      contracts.ArtifactRef `json:"artifact"`
	Replayed bool                  `json:"replayed"`
	Changed  bool                  `json:"changed"`
}

func mapRef() contracts.ArtifactRef {
	return contracts.ArtifactRef{Namespace: artifactpolicy.ProjectSitemapNamespace, Name: "map"}
}

func checkProject(ctx context.Context, db postgres.DBTX, ownerID, projectID string, lock bool) error {
	query := `SELECT lifecycle_state FROM projects WHERE owner_id=$1 AND project_id=$2`
	if lock {
		query += " FOR SHARE"
	}
	var state string
	err := db.QueryRow(ctx, query, ownerID, projectID).Scan(&state)
	if errors.Is(err, pgx.ErrNoRows) {
		return projectstore.ErrNotFound
	}
	if err != nil {
		return err
	}
	if state != "active" {
		return projectstore.ErrDeleting
	}
	return nil
}

// Read returns empty only for a missing binding, never for malformed content.
// revision selects exact retained history and never falls back to current.
func (s *Service) Read(ctx context.Context, ownerID, projectID, revision string) (View, error) {
	ctx, cancel := context.WithTimeout(ctx, OperationTimeout)
	defer cancel()
	var view View
	err := postgres.InTx(ctx, s.pool, pgx.TxOptions{AccessMode: pgx.ReadOnly}, func(tx pgx.Tx) error {
		if err := checkProject(ctx, tx, ownerID, projectID, false); err != nil {
			return err
		}
		admitted, release, err := artifacts.AcquireTransfer(ctx)
		if err != nil {
			return err
		}
		defer release()
		ctx = admitted
		store, err := artifacts.NewService(artifacts.NewPostgresRepository(tx)).Project(projectID)
		if err != nil {
			return err
		}
		ref := mapRef()
		if revision != "" {
			ref.Revision = &revision
		}
		read, err := store.Read(ctx, ref)
		if errors.Is(err, artifacts.ErrArtifactNotFound) && revision == "" {
			view = View{Document: Empty()}
			return nil
		}
		if err != nil {
			return err
		}
		if read.Payload.MediaType != MediaType {
			return ErrInvalid
		}
		document, err := Decode(read.Payload.Data)
		if err != nil {
			return err
		}
		view = View{Revision: *read.Ref.Revision, Document: document}
		return nil
	})
	return view, err
}

// Mutate reconciles retries through the existing keyed receipt, even when its
// result has been evicted. Replay does not read/recreate bytes or acquire a hold.
func (s *Service) Mutate(ctx context.Context, a Authority, idempotencyKey string, m Mutation) (Result, error) {
	if a.validate() != nil || !contracts.ValidIdempotencyKey(idempotencyKey) || len(jsonBytes(m)) > MaxBatchBytes {
		return Result{}, ErrInvalid
	}
	if m.Operation != "observe" && m.Operation != "replace-source" && m.ExpectedRevision == "" {
		return Result{}, ErrInvalid
	}
	if m.ExpectedRevision != "" && !contracts.ValidResourceID(m.ExpectedRevision) {
		return Result{}, ErrInvalid
	}
	ctx, cancel := context.WithTimeout(ctx, OperationTimeout)
	defer cancel()
	var result Result
	var err error
	for attempt := 0; attempt < MaxAttempts; attempt++ {
		err = postgres.InTx(ctx, s.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
			result = Result{}
			var inner error
			result, inner = mutateTransaction(ctx, tx, a, idempotencyKey, m)
			return inner
		})
		if errors.Is(err, postgres.ErrReceiptMismatch) {
			return Result{}, contracts.ErrIdempotencyConflict
		}
		if errors.Is(err, artifacts.ErrHistoryMaintenancePending) {
			return Result{}, artifacts.NewPostgresRepository(s.pool).MaintainHistory(ctx, err)
		}
		if err == nil || (!postgres.IsTransactionConflict(err) && !((m.Operation == "observe" || m.Operation == "replace-source" && m.ExpectedRevision == "") && errors.Is(err, artifacts.ErrArtifactConflict))) {
			return result, err
		}
		if ctx.Err() != nil {
			return Result{}, ctx.Err()
		}
	}
	return Result{}, err
}

// receiptRecord keeps only the bounded selector choices and policy conflicts
// needed to check a replay. It does not keep map history alive.
// Current grouping changes cannot reinterpret an already accepted request.
type receiptRecord struct {
	Result Result `json:"result" strictjson:"required"`
	Policy []Rule `json:"policy" strictjson:"required"`
}
type projectedRequest struct {
	Observations            []projectedObservation `json:"observations"`
	SessionGenerations      []string               `json:"sessionGenerations,omitempty"`
	ManualLocationsOmitted  bool                   `json:"manualLocationsOmitted"`
	ManualStructural        []StructuralValue      `json:"manualStructural"`
	ManualParameters        []string               `json:"manualParameters"`
	ManualStructuralOmitted bool                   `json:"manualStructuralOmitted"`
}

type projectedObservation struct {
	LocationsOmitted bool              `json:"locationsOmitted"`
	Structural       []StructuralValue `json:"structural"`
	Omitted          bool              `json:"omitted"`
}

func mutateTransaction(ctx context.Context, tx pgx.Tx, a Authority, idempotencyKey string, m Mutation) (Result, error) {
	if err := checkProject(ctx, tx, a.OwnerID, a.ProjectID, true); err != nil {
		return Result{}, err
	}
	receiptKey := postgres.ProjectSitemapReceiptKey(a.OwnerID, a.ProjectID, idempotencyKey)
	if err := postgres.LockReceipt(ctx, tx, receiptKey); err != nil {
		return Result{}, err
	}
	stored, found, err := postgres.ReadReceipt(ctx, tx, receiptKey)
	if err != nil {
		return Result{}, err
	}
	service := artifacts.NewService(artifacts.NewPostgresTxRepository(tx))
	store, err := service.Project(a.ProjectID)
	if err != nil {
		return Result{}, err
	}
	var read artifacts.ReadResult
	current := Empty()
	var policy []Rule
	if found {
		var record receiptRecord
		if err := strictjson.Decode(stored.Response, &record); err != nil {
			return Result{}, ErrInvalid
		}
		policy = record.Policy
	} else {
		admitted, release, err := artifacts.AcquireTransfer(ctx)
		if err != nil {
			return Result{}, err
		}
		defer release()
		ctx = admitted
		read, err = store.Read(ctx, mapRef())
		if err == nil {
			if read.Payload.MediaType != MediaType {
				return Result{}, ErrInvalid
			}
			current, err = Decode(read.Payload.Data)
			if err != nil {
				return Result{}, err
			}
		} else if !errors.Is(err, artifacts.ErrArtifactNotFound) {
			return Result{}, err
		}
		policy = current.Rules
	}
	intent, observations, replayPolicy, err := projectIntent(m, a, policy)
	observations.SessionGenerations = slices.Clone(m.sessionGenerations)
	if err != nil {
		return Result{}, err
	}
	digest, err := contentdigest.JSON(struct {
		Mutation   Mutation
		Projection projectedRequest
		Provenance Provenance
		Context    string
	}{intent, observations, a.provenance(), a.contextKey()})
	if err != nil {
		return Result{}, err
	}
	receipt, replayed, err := postgres.KeyedMutation(ctx, tx, receiptKey, digest, func() (postgres.Receipt, error) {
		revision := read.Ref.Revision
		actual := "absent"
		if revision != nil {
			actual = *revision
		}
		if m.ExpectedRevision != "" && m.ExpectedRevision != actual {
			return postgres.Receipt{}, ErrConflict
		}
		if a.Authorize != nil {
			if err := a.Authorize(current, m); err != nil {
				return postgres.Receipt{}, err
			}
		}
		if m.Source != nil && m.Source.Artifact != nil {
			if _, err := store.ReadExact(ctx, *m.Source.Artifact); err != nil {
				return postgres.Receipt{}, err
			}
		}
		_, data, err := Reduce(current, m, a)
		if err != nil {
			return postgres.Receipt{}, err
		}
		result := Result{Ref: read.Ref, Changed: !bytes.Equal(read.Payload.Data, data)}
		if result.Changed {
			scope, _ := artifacts.ProjectScope(a.ProjectID)
			payload := artifacts.Payload{MediaType: MediaType, Data: data}
			var written artifacts.WriteResult
			if revision == nil {
				written, err = service.Create(ctx, scope, mapRef(), payload)
			} else {
				written, err = service.Replace(ctx, scope, mapRef(), payload, *revision)
			}
			if err != nil {
				return postgres.Receipt{}, err
			}
			result.Ref = written.Ref
		}
		response, err := json.Marshal(receiptRecord{Result: result, Policy: replayPolicy})
		if len(response) > 1<<20 {
			return postgres.Receipt{}, ErrInvalid
		}
		return postgres.Receipt{OwnerID: a.OwnerID, ProjectID: a.ProjectID, ResourceID: a.ProjectID, Operation: "sitemap.mutate", Response: response}, err
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
}

// projectIntent strips removed values before receipt hashing. The saved replay
// policy preserves accepted concrete selector choices and unconditional policy
// conflicts without depending on an evictable source ref.
func projectIntent(m Mutation, a Authority, rules []Rule) (Mutation, projectedRequest, []Rule, error) {
	if len(jsonBytes(m)) > MaxBatchBytes {
		return Mutation{}, projectedRequest{}, nil, ErrInvalid
	}
	copy := m
	if m.Source != nil {
		source := *m.Source
		source.Rules = canonicalRules(m.Source.Rules)
		copy.Source = &source
	}
	copy.Observations = make([]Observation, len(m.Observations))
	statuses := make([]projectedObservation, len(m.Observations))
	replay := []Rule{}
	for i, o := range m.Observations {
		origin, path, parameters, err := projectURL(o.URL)
		status := projectedObservation{Structural: []StructuralValue{}}
		if err != nil {
			return Mutation{}, projectedRequest{}, nil, err
		}
		route, _ := routeFor(rules, origin, o.Method, path)
		values, err := projectStructural(rules, origin, o.Method, route, o.URL)
		if err != nil && !errors.Is(err, errStructuralSelection) {
			return Mutation{}, projectedRequest{}, nil, err
		}
		status.Omitted = err != nil
		status.LocationsOmitted = projectedLocations(o.URL)
		if err == nil {
			status.Structural = values
		}
		replay = addReplayPolicy(replay, rules, origin, o.Method, route, path, values, status.Omitted)
		o.URL = origin + path
		for _, name := range o.Parameters {
			if validLabel(name) {
				parameters = append(parameters, name)
			}
		}
		slices.Sort(parameters)
		o.Parameters = slices.Compact(parameters)
		copy.Observations[i] = o
		statuses[i] = status
	}
	manualStructural := []StructuralValue{}
	manualParameters := []string{}
	manualStructuralOmitted := false
	if m.URL != "" {
		origin, path, parameters, err := projectURL(m.URL)
		if err != nil {
			return Mutation{}, projectedRequest{}, nil, err
		}
		route, _ := routeFor(rules, origin, m.Method, path)
		values, err := projectStructural(rules, origin, m.Method, route, m.URL)
		if err != nil && !errors.Is(err, errStructuralSelection) {
			return Mutation{}, projectedRequest{}, nil, err
		}
		manualStructuralOmitted = err != nil
		if err == nil {
			manualStructural = values
		}
		replay = addReplayPolicy(replay, rules, origin, m.Method, route, path, manualStructural, manualStructuralOmitted)
		manualParameters = parameters
		copy.URL = origin + path
	}
	annotation, err := projectAnnotation(m.Note, m.Tags, a)
	if err != nil {
		return Mutation{}, projectedRequest{}, nil, err
	}
	copy.Note, copy.Tags = annotation.Note, annotation.Tags
	if m.Source != nil {
		if m.Source.SourceID != a.SourceID || len(m.Source.Rules) > MaxRules {
			return Mutation{}, projectedRequest{}, nil, ErrInvalid
		}
		for _, source := range []*Source{m.Source.Expected} {
			if source != nil && (!validProvenance(source.Provenance) || source.Generation == 0 || source.Generation > MaxOmissionCount || !hashPattern.MatchString(source.PolicyDigest) || source.Artifact != nil && source.Artifact.Validate() != nil) {
				return Mutation{}, projectedRequest{}, nil, ErrInvalid
			}
		}
		if m.Source.Artifact != nil {
			if m.Source.Artifact.Validate() != nil || !validText(m.Source.Artifact.Ref.Namespace) || !validText(m.Source.Artifact.Ref.Name) {
				return Mutation{}, projectedRequest{}, nil, ErrInvalid
			}
		}
		for _, rule := range m.Source.Rules {
			if validateRule(rule) != nil || !validText(rule.Path) || !validOrigin(rule.Origin) || !validText(rule.Pointer) {
				return Mutation{}, projectedRequest{}, nil, ErrInvalid
			}
			for _, name := range rule.Parameters {
				if !validLabel(name) {
					return Mutation{}, projectedRequest{}, nil, ErrInvalid
				}
			}
			for _, selector := range rule.Structural {
				if !validLabel(selector.Name) {
					return Mutation{}, projectedRequest{}, nil, ErrInvalid
				}
				for _, value := range selector.Allowed {
					if !validText(value) {
						return Mutation{}, projectedRequest{}, nil, ErrInvalid
					}
				}
			}
		}
	}
	return copy, projectedRequest{Observations: statuses, ManualLocationsOmitted: projectedLocations(m.URL), ManualStructural: manualStructural, ManualParameters: manualParameters, ManualStructuralOmitted: manualStructuralOmitted}, replay, nil
}

func addReplayPolicy(saved, rules []Rule, origin, method, route, path string, values []StructuralValue, omitted bool) []Rule {
	selectors, conflict := structuralSelectors(rules, origin, method, route)
	index := slices.IndexFunc(saved, func(r Rule) bool { return r.Origin == origin && r.Method == method && r.Path == path })
	if conflict != nil {
		// One actual conflicting pair is enough to preserve unconditional
		// omission. Keep it in one replay rule so long escaped paths are not
		// duplicated. Receipt policies may contain this duplicate selector name
		// or empty allowlists; they are projection rules, not publishable rules.
		if index < 0 {
			saved = append(saved, Rule{SourceID: "projection", Origin: origin, Method: method, Path: path, Parameters: []string{}, Structural: conflict})
		}
		return saved
	}
	if len(selectors) == 0 {
		return saved
	}
	for name := range selectors {
		selectors[name] = []string{}
	}
	if !omitted {
		for _, value := range values {
			selectors[value.Name] = append([]string{}, value.Values...)
		}
	}
	if index < 0 {
		saved = append(saved, Rule{SourceID: "projection", Origin: origin, Method: method, Path: path, Parameters: []string{}, Structural: []StructuralParameter{}})
		index = len(saved) - 1
	}
	for name, allowed := range selectors {
		position := slices.IndexFunc(saved[index].Structural, func(s StructuralParameter) bool { return s.Name == name })
		if position < 0 {
			saved[index].Structural = append(saved[index].Structural, StructuralParameter{Name: name, Allowed: []string{}})
			position = len(saved[index].Structural) - 1
		}
		selector := &saved[index].Structural[position]
		selector.Allowed = append(selector.Allowed, allowed...)
		slices.Sort(selector.Allowed)
		selector.Allowed = slices.Compact(selector.Allowed)
	}
	slices.SortFunc(saved[index].Structural, func(a, b StructuralParameter) int { return strings.Compare(a.Name, b.Name) })
	return saved
}
