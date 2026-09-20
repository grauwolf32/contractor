package runtimeconfig

import (
	"context"
	"errors"
	"sort"
	"strconv"

	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

// BindingProjection is computed from one binding/version snapshot. It is never
// stored in mutation receipts or used as authority for Run configuration.
type BindingProjection struct {
	Binding
	ToolsetConfigurationCoverage []ToolsetCoverage
	ChangedSections              []string
}

type ToolsetCoverage struct {
	Ref         string `json:"ref"`
	State       string `json:"state"`
	ValueDigest string `json:"valueDigest,omitempty"`
}

func ProjectBinding(binding Binding, version Version) (BindingProjection, error) {
	if binding.Ref != version.Ref {
		return BindingProjection{}, errors.New("Runtime label projection identity mismatch")
	}
	result := BindingProjection{Binding: binding}
	for ref, patch := range version.Spec.Worker.Toolsets {
		if !patch.Present {
			continue
		}
		coverage := ToolsetCoverage{Ref: ref, State: "cleared"}
		if !patch.Clear {
			if patch.Value == nil || patch.Value.Validate() != nil {
				return BindingProjection{}, errors.New("Runtime label projection value is invalid")
			}
			canonical, err := canonicalize(patch.Value)
			if err != nil {
				return BindingProjection{}, errors.New("Runtime label projection value is invalid")
			}
			coverage.State, coverage.ValueDigest = "configured", digest(canonical)
		}
		result.ToolsetConfigurationCoverage = append(result.ToolsetConfigurationCoverage, coverage)
	}
	sort.Slice(result.ToolsetConfigurationCoverage, func(i, j int) bool {
		return result.ToolsetConfigurationCoverage[i].Ref < result.ToolsetConfigurationCoverage[j].Ref
	})
	for section, present := range map[string]bool{
		"worker.llmGateway": version.Spec.Worker.LLMGateway.Present,
		"worker.telemetry":  version.Spec.Worker.Telemetry.Present,
		"worker.httpProxy":  version.Spec.Worker.HTTPProxy.Present,
		"worker.caido":      version.Spec.Worker.Caido.Present,
		"worker.toolsets":   len(result.ToolsetConfigurationCoverage) != 0,
		"planner.telemetry": version.Spec.Planner.Telemetry.Present,
	} {
		if present {
			result.ChangedSections = append(result.ChangedSections, section)
		}
	}
	sort.Strings(result.ChangedSections)
	return result, nil
}

// A single joined statement gives both the current revision and its immutable
// version. No endpoint/credential lookup or per-row GetVersion is performed.
const bindingProjectionSelect = `
SELECT b.label, b.config_name, b.config_version, b.config_digest, b.revision::text,
       b.created_by, b.created_at, b.updated_by, b.updated_at, v.canonical_document
FROM runtime_label_bindings b
JOIN runtime_config_versions v ON (v.name, v.version, v.digest) = (b.config_name, b.config_version, b.config_digest)`

func (r *Repository) ListBindingProjections(ctx context.Context, after string, limit int) ([]BindingProjection, error) {
	// One sentinel row supports a full 200-item public page.
	if limit < 1 || limit > maximumPageSize+1 || (after != "" && validateLabel(after) != nil) {
		return nil, invalid("RuntimeConfig binding page is invalid")
	}
	rows, err := r.db.Query(ctx, bindingProjectionSelect+` WHERE b.label > $1 ORDER BY b.label LIMIT $2`, after, limit)
	if err != nil {
		return nil, persistencepostgres.WrapError("list Runtime label coverage", err)
	}
	defer rows.Close()
	result := make([]BindingProjection, 0)
	for rows.Next() {
		item, err := scanBindingProjection(rows)
		if err != nil {
			return nil, err
		}
		result = append(result, item)
	}
	if rows.Err() != nil {
		return nil, persistencepostgres.WrapError("iterate Runtime label coverage", rows.Err())
	}
	return result, nil
}

func (r *Repository) GetBindingProjection(ctx context.Context, label string) (BindingProjection, error) {
	if err := validateLabel(label); err != nil {
		return BindingProjection{}, err
	}
	return scanBindingProjection(r.db.QueryRow(ctx, bindingProjectionSelect+` WHERE b.label = $1`, label))
}

func scanBindingProjection(row rowScanner) (BindingProjection, error) {
	var binding Binding
	var revision, canonical string
	err := row.Scan(&binding.Label, &binding.Ref.Name, &binding.Ref.Version, &binding.Ref.Digest,
		&revision, &binding.CreatedBy, &binding.CreatedAt, &binding.UpdatedBy, &binding.UpdatedAt, &canonical)
	if errors.Is(err, pgx.ErrNoRows) {
		return BindingProjection{}, ErrNotFound
	}
	if err != nil {
		return BindingProjection{}, persistencepostgres.WrapError("read Runtime label coverage", err)
	}
	binding.Revision, err = strconv.ParseUint(revision, 10, 64)
	if err != nil || binding.Revision == 0 {
		return BindingProjection{}, errors.New("stored Runtime label revision is invalid")
	}
	version, err := DecodeStoredDocument([]byte(canonical))
	if err != nil {
		return BindingProjection{}, errors.New("stored Runtime label version failed integrity validation")
	}
	return ProjectBinding(binding, version)
}
