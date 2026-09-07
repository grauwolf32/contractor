package e2e

import (
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"go.yaml.in/yaml/v4"
)

type performanceMetricsMatrix struct {
	SchemaVersion string                         `yaml:"schema_version"`
	Policy        performanceMetricsMatrixPolicy `yaml:"policy"`
	Cases         []performanceMetricsMatrixCase `yaml:"cases"`
	Gates         []matrixGate                   `yaml:"gates"`
}

type performanceMetricsMatrixPolicy struct {
	Switches  string `yaml:"switches"`
	Sampling  string `yaml:"sampling"`
	Authority string `yaml:"authority"`
	Evidence  string `yaml:"evidence"`
}

type performanceMetricsMatrixCase struct {
	ID         string                     `yaml:"id"`
	Group      string                     `yaml:"group"`
	Acceptance []string                   `yaml:"acceptance"`
	Faults     []string                   `yaml:"faults"`
	Expected   string                     `yaml:"expected"`
	Test       lifecycleControlsTestOwner `yaml:"test"`
}

var (
	performanceMetricsAcceptance = []string{"A1", "A2", "A3", "A4"}
	performanceMetricsGroups     = []string{
		"startup_switches", "cadence_history", "http_accounting", "postgres_diagnostics",
		"allocation_resources", "authenticated_operations", "profiling", "overhead_evidence",
	}
	performanceMetricsFaults = []string{
		"F01", "F02", "F03", "F04", "F05", "F06", "F07", "F08", "F09", "F10",
		"F11", "F12", "F13", "F14", "F15", "F16", "F17", "F18", "F19", "F20",
		"F21", "F22", "F23", "F24", "F25", "F26", "F27", "F28", "F29", "F30",
		"F31", "F32", "F33", "F34", "F35",
	}
)

func TestPerformanceMetricsMatrixIsComplete(t *testing.T) {
	repositoryRoot := filepath.Join("..", "..")
	data, err := os.ReadFile("performance_matrix.yml")
	if err != nil {
		t.Fatal(err)
	}
	matrix, err := decodeStrictMatrix[performanceMetricsMatrix](data, "performance metrics")
	if err != nil {
		t.Fatal(err)
	}
	if err := validatePerformanceMetricsMatrix(repositoryRoot, matrix); err != nil {
		t.Fatal(err)
	}

	t.Run("missing policy", func(t *testing.T) {
		broken := mustDecodeStrictMatrix[performanceMetricsMatrix](t, data, "performance metrics")
		broken.Policy.Authority = ""
		requirePerformanceMetricsMatrixError(t, repositoryRoot, broken, "policy")
	})
	t.Run("missing acceptance", func(t *testing.T) {
		broken := mustDecodeStrictMatrix[performanceMetricsMatrix](t, data, "performance metrics")
		for index := range broken.Cases {
			broken.Cases[index].Acceptance = slices.DeleteFunc(
				broken.Cases[index].Acceptance, func(value string) bool { return value == "A3" },
			)
		}
		requirePerformanceMetricsMatrixError(t, repositoryRoot, broken, "A3")
	})
	t.Run("missing group", func(t *testing.T) {
		broken := mustDecodeStrictMatrix[performanceMetricsMatrix](t, data, "performance metrics")
		broken.Cases = slices.DeleteFunc(
			broken.Cases, func(value performanceMetricsMatrixCase) bool { return value.Group == "profiling" },
		)
		requirePerformanceMetricsMatrixError(t, repositoryRoot, broken, `group "profiling"`)
	})
	t.Run("missing fault", func(t *testing.T) {
		broken := mustDecodeStrictMatrix[performanceMetricsMatrix](t, data, "performance metrics")
		for index := range broken.Cases {
			broken.Cases[index].Faults = slices.DeleteFunc(
				broken.Cases[index].Faults, func(value string) bool { return value == "F34" },
			)
		}
		requirePerformanceMetricsMatrixError(t, repositoryRoot, broken, "F34")
	})
	t.Run("renamed owner", func(t *testing.T) {
		broken := mustDecodeStrictMatrix[performanceMetricsMatrix](t, data, "performance metrics")
		broken.Cases[0].Test.Name += "Renamed"
		requirePerformanceMetricsMatrixError(t, repositoryRoot, broken, "does not define")
	})
	t.Run("missing benchmark gate", func(t *testing.T) {
		broken := mustDecodeStrictMatrix[performanceMetricsMatrix](t, data, "performance metrics")
		broken.Gates = slices.DeleteFunc(broken.Gates, func(value matrixGate) bool {
			return value.ID == "benchmark"
		})
		requirePerformanceMetricsMatrixError(t, repositoryRoot, broken, `gate "benchmark"`)
	})
}

func requirePerformanceMetricsMatrixError(
	t *testing.T,
	repositoryRoot string,
	matrix performanceMetricsMatrix,
	want string,
) {
	t.Helper()
	err := validatePerformanceMetricsMatrix(repositoryRoot, matrix)
	if err == nil || !strings.Contains(err.Error(), want) {
		t.Fatalf("matrix error = %v, want %q", err, want)
	}
}

func validatePerformanceMetricsMatrix(repositoryRoot string, matrix performanceMetricsMatrix) error {
	if matrix.SchemaVersion != "1.0" || strings.TrimSpace(matrix.Policy.Switches) == "" ||
		strings.TrimSpace(matrix.Policy.Sampling) == "" || strings.TrimSpace(matrix.Policy.Authority) == "" ||
		strings.TrimSpace(matrix.Policy.Evidence) == "" {
		return fmt.Errorf("incomplete performance-metrics policy: %+v", matrix.Policy)
	}
	seenCases := map[string]bool{}
	seenAcceptance := map[string]bool{}
	seenGroups := map[string]bool{}
	seenFaults := map[string]bool{}
	seenOwners := map[string]string{}
	for _, item := range matrix.Cases {
		if item.ID == "" || item.Group == "" || strings.TrimSpace(item.Expected) == "" ||
			seenCases[item.ID] || len(item.Acceptance) == 0 {
			return fmt.Errorf("invalid or duplicate performance-metrics case: %+v", item)
		}
		if !slices.Contains(performanceMetricsGroups, item.Group) {
			return fmt.Errorf("case %q has unknown group %q", item.ID, item.Group)
		}
		seenCases[item.ID] = true
		seenGroups[item.Group] = true
		for _, acceptance := range item.Acceptance {
			if !slices.Contains(performanceMetricsAcceptance, acceptance) {
				return fmt.Errorf("case %q has unknown acceptance %q", item.ID, acceptance)
			}
			seenAcceptance[acceptance] = true
		}
		for _, fault := range item.Faults {
			if !slices.Contains(performanceMetricsFaults, fault) || seenFaults[fault] {
				return fmt.Errorf("case %q has unknown or duplicate fault %q", item.ID, fault)
			}
			seenFaults[fault] = true
		}
		owner := item.Test.Source + "#" + item.Test.Kind + "#" + item.Test.Name
		if previous, exists := seenOwners[owner]; exists {
			return fmt.Errorf("duplicate test owner %q in cases %q and %q", owner, previous, item.ID)
		}
		seenOwners[owner] = item.ID
		if err := validateMatrixNamedOwner(
			repositoryRoot, item.Test.Source, item.Test.Kind, item.Test.Name,
		); err != nil {
			return fmt.Errorf("case %q: %w", item.ID, err)
		}
	}
	for _, acceptance := range performanceMetricsAcceptance {
		if !seenAcceptance[acceptance] {
			return fmt.Errorf("required performance-metrics acceptance %q is absent", acceptance)
		}
	}
	for _, group := range performanceMetricsGroups {
		if !seenGroups[group] {
			return fmt.Errorf("required performance-metrics group %q is absent", group)
		}
	}
	for _, fault := range performanceMetricsFaults {
		if !seenFaults[fault] {
			return fmt.Errorf("required performance-metrics fault %q is absent", fault)
		}
	}
	return validateMatrixGates(
		"performance metrics", matrix.Gates,
		[]string{"benchmark", "browser", "go", "matrix", "postgres", "release", "runtime"}, true,
	)
}

type performanceReleaseEvidence struct {
	SchemaVersion string `yaml:"schema_version"`
	Environment   struct {
		RecordedAt string `yaml:"recorded_at"`
		Go         string `yaml:"go"`
		OS         string `yaml:"os"`
		Arch       string `yaml:"arch"`
		CPU        string `yaml:"cpu"`
		LogicalCPU int    `yaml:"logical_cpu"`
		Memory     string `yaml:"memory"`
	} `yaml:"environment"`
	Harness struct {
		Command     string `yaml:"command"`
		Revision    string `yaml:"revision"`
		Repetitions int    `yaml:"repetitions"`
	} `yaml:"harness"`
	Scenarios []struct {
		Name     string `yaml:"name"`
		Workload string `yaml:"workload"`
		Samples  []struct {
			Operations          int     `yaml:"operations"`
			WallSeconds         float64 `yaml:"wall_seconds"`
			CPUSeconds          float64 `yaml:"cpu_seconds"`
			RSSBytes            uint64  `yaml:"rss_bytes"`
			AllocationsPerOp    float64 `yaml:"allocations_per_op"`
			ThroughputPerSecond float64 `yaml:"throughput_per_second"`
			P95Nanoseconds      float64 `yaml:"p95_nanoseconds"`
			RetainedFrames      int     `yaml:"retained_frames"`
			RetainedBytes       int     `yaml:"retained_bytes"`
			PendingMinutes      int     `yaml:"pending_minutes"`
			DatabaseReads       uint64  `yaml:"database_reads"`
			DatabaseSizeReads   uint64  `yaml:"database_size_reads"`
			HistoryWrites       uint64  `yaml:"history_writes"`
			ProfileBytes        uint64  `yaml:"profile_response_bytes"`
		} `yaml:"samples"`
		Summary struct {
			ThroughputMean  float64 `yaml:"throughput_mean"`
			ThroughputCV    float64 `yaml:"throughput_cv"`
			P95Mean         float64 `yaml:"p95_mean_nanoseconds"`
			P95CV           float64 `yaml:"p95_cv"`
			CPUMean         float64 `yaml:"cpu_mean_seconds"`
			RSSMean         float64 `yaml:"rss_mean_bytes"`
			AllocationsMean float64 `yaml:"allocations_mean_per_op"`
		} `yaml:"summary"`
	} `yaml:"scenarios"`
	Interpretation []string `yaml:"interpretation"`
}

func TestPerformanceReleaseEvidenceIsComplete(t *testing.T) {
	data, err := os.ReadFile(filepath.Join("..", "..", "docs", "reviews", "2026-09-06-performance-v32-release.yml"))
	if err != nil {
		t.Fatal(err)
	}
	var evidence performanceReleaseEvidence
	decoder := yaml.NewDecoder(strings.NewReader(string(data)))
	decoder.KnownFields(true)
	if err := decoder.Decode(&evidence); err != nil {
		t.Fatal(err)
	}
	if evidence.SchemaVersion != "1.0" || evidence.Environment.RecordedAt == "" ||
		evidence.Environment.Go == "" || evidence.Environment.OS == "" || evidence.Environment.Arch == "" ||
		evidence.Environment.CPU == "" || evidence.Environment.LogicalCPU < 1 || evidence.Environment.Memory == "" ||
		!strings.HasPrefix(evidence.Harness.Command, "make benchmark-performance") ||
		evidence.Harness.Revision == "" || evidence.Harness.Repetitions < 5 || len(evidence.Interpretation) < 2 {
		t.Fatalf("incomplete release evidence metadata: %+v", evidence)
	}
	required := map[string]bool{
		"http_disabled": false, "http_enabled": false,
		"collection_disabled": false, "collection_enabled": false,
		"pprof_off": false, "pprof_idle": false, "pprof_active": false,
	}
	for _, scenario := range evidence.Scenarios {
		if _, exists := required[scenario.Name]; !exists || required[scenario.Name] ||
			strings.TrimSpace(scenario.Workload) == "" || len(scenario.Samples) != evidence.Harness.Repetitions ||
			scenario.Summary.ThroughputMean <= 0 || scenario.Summary.ThroughputCV < 0 ||
			scenario.Summary.P95Mean < 0 || scenario.Summary.P95CV < 0 ||
			scenario.Summary.CPUMean < 0 || scenario.Summary.RSSMean <= 0 || scenario.Summary.AllocationsMean < 0 {
			t.Fatalf("invalid release evidence scenario: %+v", scenario)
		}
		required[scenario.Name] = true
		for _, sample := range scenario.Samples {
			if sample.Operations < 1 || sample.WallSeconds <= 0 || sample.CPUSeconds < 0 || sample.RSSBytes == 0 ||
				sample.AllocationsPerOp < 0 || sample.ThroughputPerSecond <= 0 || sample.P95Nanoseconds < 0 {
				t.Fatalf("invalid release evidence sample for %s: %+v", scenario.Name, sample)
			}
		}
	}
	for name, found := range required {
		if !found {
			t.Fatalf("required release evidence scenario %q is absent", name)
		}
	}
	for _, scenario := range evidence.Scenarios {
		switch scenario.Name {
		case "collection_disabled":
			for _, sample := range scenario.Samples {
				if sample.RetainedFrames != 0 || sample.RetainedBytes != 0 || sample.PendingMinutes != 0 ||
					sample.DatabaseReads != 0 || sample.DatabaseSizeReads != 0 || sample.HistoryWrites != 0 {
					t.Fatalf("disabled collection performed retained or diagnostic work: %+v", sample)
				}
			}
		case "collection_enabled":
			for _, sample := range scenario.Samples {
				if sample.RetainedFrames != 240 || sample.RetainedBytes <= 0 || sample.RetainedBytes > 8*1024*1024 ||
					sample.PendingMinutes > 10 || sample.DatabaseReads != 60 ||
					sample.DatabaseSizeReads != 12 || sample.HistoryWrites != 60 {
					t.Fatalf("enabled collection exceeded fixed state or logical IO: %+v", sample)
				}
			}
		case "pprof_active":
			for _, sample := range scenario.Samples {
				if sample.ProfileBytes == 0 {
					t.Fatalf("active profiling sample has no readable response: %+v", sample)
				}
			}
		}
	}
}
