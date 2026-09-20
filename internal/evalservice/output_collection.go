package evalservice

import (
	"context"
	"slices"
	"sort"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/grauwolf32/contractor/internal/evalstore"
	pg "github.com/grauwolf32/contractor/internal/persistence/postgres"
)

// OutputObservation keeps imported exact refs only. Normalization never reads
// private expected findings or rubric text to construct execution evidence.
type OutputObservation struct {
	Outputs    map[string]evaldomain.Artifact
	Evidence   []evaldomain.Evidence
	Collection evaldomain.Collection
}

// Preparation and collection must agree on the exact Audit artifact behind
// each output. Coverage/findings retain the machine report with a JSON pointer;
// the Markdown report is selected explicitly through the summary output.
var auditOutputContracts = map[string]struct {
	logicalKey string
	mediaType  string
}{
	"report":   {auditstore.ReportMachineLogicalKey, "application/json"},
	"coverage": {auditstore.ReportMachineLogicalKey, "application/json"},
	"findings": {auditstore.ReportMachineLogicalKey, "application/json"},
	"summary":  {auditstore.ReportSummaryLogicalKey, "text/markdown"},
}

func collectOutputs(
	ctx context.Context,
	db pg.DBTX,
	owner string,
	member evalstore.Member,
	execution ExecutionView,
	inventory evalstore.Inventory,
) (OutputObservation, error) {
	observation := OutputObservation{
		Outputs:    map[string]evaldomain.Artifact{},
		Evidence:   []evaldomain.Evidence{},
		Collection: evaldomain.Collection{Status: "unavailable", Gaps: []string{}},
	}
	if execution.Ref == nil {
		observation.Collection.Gaps = append(observation.Collection.Gaps, "Execution has not been confirmed.")
		return observation, nil
	}
	if !isTerminal(execution.State) {
		observation.Collection.Gaps = append(observation.Collection.Gaps, "Execution is not terminal.")
	}
	observation.Collection.Gaps = append(observation.Collection.Gaps, inventory.Gaps...)

	store := evalstore.NewPostgresStore(db)
	for _, role := range outputRoles(member.Recipe.Case.Outputs) {
		contract := member.Recipe.Case.Outputs[role]
		slot := mappedOutput(member.Recipe.Variant, role)
		ref, err := resolveOutput(ctx, store, owner, *execution.Ref, slot)
		if evaldomain.IsCode(err, "eval_not_found") || evaldomain.IsCode(err, "eval_evidence_unavailable") {
			if contract.Required {
				observation.Collection.Gaps = append(observation.Collection.Gaps, "Required output is unavailable: "+role)
			}
			continue
		}
		if err != nil {
			return observation, err
		}
		if !slices.Contains(contract.MediaTypes, ref.MediaType) {
			observation.Collection.Gaps = append(observation.Collection.Gaps, "Output media type differs from its contract: "+role)
		}

		observation.Outputs[role] = ref
		evidence := evaldomain.Evidence{ID: role, Artifact: ref}
		if execution.Ref.Kind == "audit" && (slot == "coverage" || slot == "findings") {
			evidence.Location = "/" + slot
		}
		observation.Evidence = append(observation.Evidence, evidence)
	}

	observation.Collection.Gaps = evaldomain.BoundedGaps(observation.Collection.Gaps)
	switch {
	case len(observation.Collection.Gaps) == 0:
		observation.Collection.Status = "complete"
	case len(observation.Outputs) > 0:
		observation.Collection.Status = "partial"
	}
	return observation, nil
}

func outputRoles(outputs map[string]evaldomain.Output) []string {
	roles := make([]string, 0, len(outputs))
	for role := range outputs {
		roles = append(roles, role)
	}
	sort.Strings(roles)
	return roles
}

func mappedOutput(variant evaldomain.Variant, role string) string {
	if slot, exists := variant.OutputMapping[role]; exists {
		return slot
	}
	return role
}

func resolveOutput(
	ctx context.Context,
	store *evalstore.Store,
	owner string,
	execution evaldomain.ExecutionRef,
	slot string,
) (evaldomain.Artifact, error) {
	if execution.Kind == "run" {
		return store.RunOutput(ctx, owner, execution.ID, slot)
	}

	output, ok := auditOutputContracts[slot]
	if !ok {
		return evaldomain.Artifact{}, evaldomain.Failure("eval_not_found")
	}
	return store.AuditOutput(ctx, owner, execution.ID, output.logicalKey)
}
