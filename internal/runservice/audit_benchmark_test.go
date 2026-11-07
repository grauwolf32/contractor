package runservice

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
)

// BenchmarkAuditRunCreatePinnedProgram exercises ordinary child Run creation
// through real PostgreSQL stores. Rolling back only after the full callback
// succeeds lets every iteration reuse the same accepted intent without growing
// retained business data or measuring fixture setup.
func BenchmarkAuditRunCreatePinnedProgram(b *testing.B) {
	for _, instructionBytes := range []int{0, 64 * 1024} {
		b.Run(fmt.Sprintf("instructions-%d", instructionBytes), func(b *testing.B) {
			b.StopTimer()
			fixture := newCommonTaskRunFixtureWithProgram(b, benchmarkAuditProgram(), true, func(workflow *resolved.ResolvedWorkflow) {
				if instructionBytes == 0 {
					return
				}
				stage := workflow.Stages["check"]
				const instruction = "Inspect the selected exact source. "
				stage.Instructions.Text = strings.Repeat(instruction, instructionBytes/len(instruction)+1)[:instructionBytes]
				stage.Instructions.Digest = contentdigest.Bytes([]byte(stage.Instructions.Text))
				workflow.Stages["check"] = stage
			})
			transaction := fixture.service.auditTransaction
			rolledBack := errors.New("benchmark creation rolled back")
			fixture.service.auditTransaction = func(ctx context.Context, create func(AuditRunWriter, *artifacts.Service, AuditExecutionWriter) error) error {
				return transaction(ctx, func(runs AuditRunWriter, artifacts *artifacts.Service, audits AuditExecutionWriter) error {
					if err := create(runs, artifacts, audits); err != nil {
						return err
					}
					return rolledBack
				})
			}
			b.ReportAllocs()
			b.ResetTimer()
			b.StartTimer()
			for b.Loop() {
				if _, err := fixture.service.CreateAudit(b.Context(), fixture.params); !errors.Is(err, rolledBack) {
					b.Fatalf("create did not reach its final dispatch fence: %v", err)
				}
			}
			b.ReportMetric(float64(len(fixture.audit.ProfileSnapshot)), "pin-B")
		})
	}
}

// The benchmark authors its own bounded program: shared cross-language
// fixtures belong to TestShared* verdicts and are not benchmark inputs.
func benchmarkAuditProgram() auditcontract.Program {
	return auditcontract.Program{
		Schema: "contractor.audit.program.v1",
		Inputs: map[string]auditcontract.ProgramInput{
			"source": {Required: true, MediaTypes: []string{"text/plain"}},
		},
		Workflows: map[string]auditcontract.ProgramWorkflow{
			"inspect": {
				Ref: "artifact-copy@1", Inputs: map[string]auditcontract.InputMapping{
					"source": {Source: "audit-input", Name: "source"},
				},
				Parameters: map[string]string{}, Outputs: map[string]string{"material": "result"},
			},
		},
		Hypotheses:  map[string]auditcontract.HypothesisSeed{},
		InitialWork: []auditcontract.InitialWork{{Key: "inspect-source", Workflow: "inspect", Objective: "Inspect the selected source"}},
		Limits: auditcontract.ProgramLimits{
			MaxTasks: 128, MaxIterations: 4, MaxTasksPerIteration: 128,
			BatchSize: 2, MaxSubmittedRuns: 512, MaxAttempts: 3, DeadlineSeconds: 86400,
			MaxEvidenceBytes: 64 << 20, MaxHypotheses: 128, MaxHypothesisRevisions: 256, MaxHypothesisBytes: 1 << 20,
		},
		Policy: auditcontract.OperationPolicy{ActiveOperations: "prohibited"},
	}
}
