package evalservice

import (
	"context"
	"encoding/json"
	"reflect"
	"slices"

	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/grauwolf32/contractor/internal/evalstore"
	pg "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

type ViewMetadata struct {
	Snapshot  string             `json:"viewSnapshot"`
	Freshness string             `json:"freshness"`
	Summary   evaldomain.Summary `json:"experimentSummary"`
}

func validPage(after, limit int) bool {
	return after >= -1 && limit >= 1 && limit <= evaldomain.MaxPageSize
}

func selectedPageParams(p MemberPageParams, v evalstore.View) evalstore.SelectedPageParams {
	return evalstore.SelectedPageParams{
		OwnerID:      p.OwnerID,
		ExperimentID: p.ExperimentID,
		Generation:   v.Generation,
		SuiteID:      p.SuiteID,
		VariantID:    p.VariantID,
		Filter:       p.Filter,
		AfterOrdinal: p.AfterOrdinal,
		Limit:        p.Limit,
		Bin:          p.Bin,
	}
}

func (s *Service) withSelectedView(ctx context.Context, p MemberPageParams, read func(*evalstore.Store, evalstore.Experiment, evalstore.View, preparedSetup) error) error {
	return pg.InTx(ctx, s.pool, pgx.TxOptions{IsoLevel: pgx.RepeatableRead, AccessMode: pgx.ReadOnly}, func(tx pgx.Tx) error {
		st := evalstore.NewTxStore(tx)
		e, err := st.Get(ctx, p.OwnerID, p.ExperimentID)
		if err != nil {
			return err
		}
		view, err := st.LatestView(ctx, p.OwnerID, p.ExperimentID)
		if err != nil {
			return err
		}
		if view == nil {
			return evaldomain.Failure("eval_not_ready")
		}
		if p.Snapshot != "" && p.Snapshot != view.Snapshot {
			return evaldomain.Failure("eval_view_changed")
		}
		if p.SuiteID != "" {
			if _, ok := view.Suites[p.SuiteID]; !ok {
				return evaldomain.Failure("eval_invalid")
			}
		}
		plan, err := st.FrozenPlan(ctx, p.OwnerID, p.ExperimentID)
		if err != nil {
			return err
		}
		var setup preparedSetup
		if err = json.Unmarshal(plan.Setup, &setup); err != nil {
			return err
		}
		if len(setup.Variants) != 2 {
			return evaldomain.Failure("eval_member_conflict")
		}
		if p.MeasurementScope != "" && p.MeasurementScope != setup.Variants[0].Kind {
			return evaldomain.Failure("eval_invalid")
		}
		if p.Bin != nil && p.Bin.Scope != setup.Variants[0].Kind {
			return evaldomain.Failure("eval_invalid")
		}
		return read(st, e, *view, setup)
	})
}

type PairPageParams struct {
	MemberPageParams
	Metric          string
	Absolute        bool
	AfterDifference *float64
}

type PairPage struct {
	ViewMetadata
	evalstore.SelectedPage[evaldomain.Pair]
}

func (s *Service) Pairs(ctx context.Context, p PairPageParams) (PairPage, error) {
	var out PairPage
	if !validPage(p.AfterOrdinal, p.Limit) || !slices.Contains([]string{"", "all", "regressions", "unresolved"}, p.Filter) || p.VariantID != "" {
		return out, evaldomain.Failure("eval_invalid")
	}
	err := s.withSelectedView(ctx, p.MemberPageParams, func(st *evalstore.Store, e evalstore.Experiment, v evalstore.View, setup preparedSetup) error {
		params := selectedPageParams(p.MemberPageParams, v)
		params.Metric, params.Absolute, params.AfterDifference = p.Metric, p.Absolute, p.AfterDifference
		var err error
		out.SelectedPage, err = st.PairPage(ctx, params)
		out.ViewMetadata = ViewMetadata{v.Snapshot, v.Freshness, v.Summary}
		return err
	})
	return out, err
}

type AttributedRecord struct {
	evalstore.Record
	Document json.RawMessage `json:"document"`
}
type PairDetail struct {
	ViewMetadata
	Pair    evaldomain.Pair    `json:"pair"`
	Records []AttributedRecord `json:"records"`
}

func (s *Service) Pair(ctx context.Context, p MemberPageParams, pairID string) (PairDetail, error) {
	out := PairDetail{Records: []AttributedRecord{}}
	err := s.withSelectedView(ctx, p, func(st *evalstore.Store, e evalstore.Experiment, v evalstore.View, setup preparedSetup) error {
		pair, err := st.ViewPair(ctx, e.OwnerID, e.ID, v.Generation, pairID)
		if err != nil {
			return err
		}
		out.Pair = pair
		out.ViewMetadata = ViewMetadata{v.Snapshot, v.Freshness, v.Summary}
		for _, member := range []evaldomain.MemberView{pair.A, pair.B} {
			records := []struct {
				kind   string
				digest *string
			}{
				{evaldomain.RecordKindResult, member.ResultSHA256},
				{evaldomain.RecordKindAssessment, member.AssessmentSHA256},
			}
			for _, selected := range records {
				if selected.digest == nil {
					continue
				}
				record, err := st.Record(ctx, e.OwnerID, e.ID, member.Member.ID, selected.kind, *selected.digest)
				if err != nil {
					return err
				}
				out.Records = append(out.Records, AttributedRecord{Record: record, Document: record.Document.Bytes()})
			}
		}
		return nil
	})
	return out, err
}

type Report struct {
	SchemaVersion string              `json:"schemaVersion"`
	ExperimentID  string              `json:"experimentId"`
	PlanSHA256    string              `json:"planSha256"`
	Snapshot      string              `json:"viewSnapshot"`
	Summary       evaldomain.Summary  `json:"summary"`
	Sources       []evaldomain.Source `json:"sources"`
	PairCount     int                 `json:"pairCount"`
}

func (s *Service) Report(ctx context.Context, p MemberPageParams) (Report, error) {
	out := Report{SchemaVersion: "contractor.eval-report/v1", Sources: []evaldomain.Source{}}
	err := s.withSelectedView(ctx, p, func(st *evalstore.Store, e evalstore.Experiment, v evalstore.View, setup preparedSetup) error {
		plan, err := st.FrozenPlan(ctx, e.OwnerID, e.ID)
		if err != nil {
			return err
		}
		out.ExperimentID, out.PlanSHA256, out.Snapshot, out.Summary, out.PairCount = e.ID, plan.SHA256, v.Snapshot, v.Summary, e.Expected/2
		// Registration provenance is safe. Private rubric bodies and check reasons
		// never enter exports, including human-entered text.
		var metadata struct {
			Source *evaldomain.Source `json:"source"`
		}
		if err = json.Unmarshal(plan.Setup, &metadata); err != nil {
			return err
		}
		out.Sources, err = st.ViewSources(ctx, e.OwnerID, e.ID, v.Generation)
		if err != nil {
			return err
		}
		if metadata.Source != nil && !slices.ContainsFunc(out.Sources, func(source evaldomain.Source) bool { return reflect.DeepEqual(source, *metadata.Source) }) {
			out.Sources = append(out.Sources, *metadata.Source)
		}
		if len(out.Sources) > evaldomain.MaxReportSources {
			return evaldomain.Failure("eval_limit_exceeded")
		}
		return nil
	})
	return out, err
}
