// Command performancebench records reproducible local evidence for V32.
// It is a measurement harness, not a universal performance acceptance limit.
package main

import (
	"bufio"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	contractorperformance "github.com/grauwolf32/contractor/internal/performance"
	contractorprofiling "github.com/grauwolf32/contractor/internal/profiling"
	"golang.org/x/sys/unix"
)

const schemaVersion = "contractor.performance-benchmark.v1"

type measurement struct {
	Scenario             string  `json:"scenario"`
	Operations           int     `json:"operations"`
	WallSeconds          float64 `json:"wallSeconds"`
	CPUSeconds           float64 `json:"cpuSeconds"`
	RSSBytes             uint64  `json:"rssBytes"`
	AllocationsPerOp     float64 `json:"allocationsPerOp"`
	ThroughputPerSecond  float64 `json:"throughputPerSecond"`
	P95Nanoseconds       float64 `json:"p95Nanoseconds"`
	RetainedFrames       int     `json:"retainedFrames,omitempty"`
	RetainedBytes        int     `json:"retainedBytes,omitempty"`
	PendingMinutes       int     `json:"pendingMinutes,omitempty"`
	DatabaseReads        uint64  `json:"databaseReads,omitempty"`
	DatabaseSizeReads    uint64  `json:"databaseSizeReads,omitempty"`
	HistoryWrites        uint64  `json:"historyWrites,omitempty"`
	ProfileResponseBytes uint64  `json:"profileResponseBytes,omitempty"`
}

type scenarioReport struct {
	Name    string        `json:"name"`
	Samples []measurement `json:"samples"`
	Summary summary       `json:"summary"`
}

type summary struct {
	ThroughputMean  float64 `json:"throughputMean"`
	ThroughputCV    float64 `json:"throughputCV"`
	P95Mean         float64 `json:"p95MeanNanoseconds"`
	P95CV           float64 `json:"p95CV"`
	CPUMean         float64 `json:"cpuMeanSeconds"`
	RSSMean         float64 `json:"rssMeanBytes"`
	AllocationsMean float64 `json:"allocationsMeanPerOp"`
}

type benchmarkReport struct {
	SchemaVersion string           `json:"schemaVersion"`
	GeneratedAt   string           `json:"generatedAt"`
	GoVersion     string           `json:"goVersion"`
	OS            string           `json:"os"`
	Architecture  string           `json:"architecture"`
	LogicalCPU    int              `json:"logicalCPU"`
	Scenarios     []scenarioReport `json:"scenarios"`
}

type options struct {
	child            bool
	scenario         string
	repetitions      int
	httpRequests     int
	collectionCycles int
	profileDuration  time.Duration
}

func main() {
	var configuration options
	var profileSeconds int
	flag.BoolVar(&configuration.child, "child", false, "run one isolated child measurement")
	flag.StringVar(&configuration.scenario, "scenario", "", "isolated child scenario")
	flag.IntVar(&configuration.repetitions, "repetitions", 5, "isolated repetitions per scenario")
	flag.IntVar(&configuration.httpRequests, "http-requests", 50_000, "HTTP operations per repetition")
	flag.IntVar(&configuration.collectionCycles, "collection-cycles", 240, "15-second collection frames per repetition")
	flag.IntVar(&profileSeconds, "profile-seconds", 1, "busy workload seconds for each profiling variant")
	flag.Parse()
	configuration.profileDuration = time.Duration(profileSeconds) * time.Second
	if err := validateOptions(configuration); err != nil {
		fatal(err)
	}
	if configuration.child {
		result, err := runScenario(configuration)
		if err != nil {
			fatal(err)
		}
		if err := json.NewEncoder(os.Stdout).Encode(result); err != nil {
			fatal(err)
		}
		return
	}
	report, err := runParent(configuration)
	if err != nil {
		fatal(err)
	}
	encoder := json.NewEncoder(os.Stdout)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(report); err != nil {
		fatal(err)
	}
}

func validateOptions(configuration options) error {
	if configuration.repetitions < 1 || configuration.repetitions > 20 ||
		configuration.httpRequests < 1_000 || configuration.httpRequests > 2_000_000 ||
		configuration.collectionCycles < 20 || configuration.collectionCycles > 10_000 ||
		configuration.profileDuration < time.Second || configuration.profileDuration > 10*time.Second {
		return errors.New("benchmark bounds are invalid")
	}
	return nil
}

func runParent(configuration options) (benchmarkReport, error) {
	executable, err := os.Executable()
	if err != nil {
		return benchmarkReport{}, err
	}
	names := []string{
		"http_disabled", "http_enabled",
		"collection_disabled", "collection_enabled",
		"pprof_off", "pprof_idle", "pprof_active",
	}
	report := benchmarkReport{
		SchemaVersion: schemaVersion,
		GeneratedAt:   time.Now().UTC().Format(time.RFC3339),
		GoVersion:     runtime.Version(),
		OS:            runtime.GOOS,
		Architecture:  runtime.GOARCH,
		LogicalCPU:    runtime.NumCPU(),
		Scenarios:     make([]scenarioReport, 0, len(names)),
	}
	for _, name := range names {
		item := scenarioReport{Name: name, Samples: make([]measurement, 0, configuration.repetitions)}
		for range configuration.repetitions {
			command := exec.Command(
				executable,
				"-child", "-scenario", name,
				"-repetitions", strconv.Itoa(configuration.repetitions),
				"-http-requests", strconv.Itoa(configuration.httpRequests),
				"-collection-cycles", strconv.Itoa(configuration.collectionCycles),
				"-profile-seconds", strconv.Itoa(int(configuration.profileDuration/time.Second)),
			)
			output, childErr := command.Output()
			if childErr != nil {
				var exitError *exec.ExitError
				if errors.As(childErr, &exitError) {
					return benchmarkReport{}, fmt.Errorf("%s child: %w: %s", name, childErr, exitError.Stderr)
				}
				return benchmarkReport{}, fmt.Errorf("%s child: %w", name, childErr)
			}
			var value measurement
			if err := json.Unmarshal(output, &value); err != nil {
				return benchmarkReport{}, fmt.Errorf("decode %s child: %w", name, err)
			}
			if value.Scenario != name || value.Operations < 1 || value.WallSeconds <= 0 ||
				value.ThroughputPerSecond <= 0 || value.RSSBytes == 0 {
				return benchmarkReport{}, fmt.Errorf("invalid %s measurement: %+v", name, value)
			}
			item.Samples = append(item.Samples, value)
		}
		item.Summary = summarize(item.Samples)
		report.Scenarios = append(report.Scenarios, item)
	}
	return report, nil
}

func runScenario(configuration options) (measurement, error) {
	switch configuration.scenario {
	case "http_disabled":
		return measureHTTP(false, configuration.httpRequests), nil
	case "http_enabled":
		return measureHTTP(true, configuration.httpRequests), nil
	case "collection_disabled":
		return measureCollection(false, configuration.collectionCycles), nil
	case "collection_enabled":
		return measureCollection(true, configuration.collectionCycles), nil
	case "pprof_off":
		return measureProfiling("pprof_off", configuration.profileDuration)
	case "pprof_idle":
		return measureProfiling("pprof_idle", configuration.profileDuration)
	case "pprof_active":
		return measureProfiling("pprof_active", configuration.profileDuration)
	default:
		return measurement{}, errors.New("unknown child scenario")
	}
}

type responseSink struct {
	header http.Header
	status int
}

func (w *responseSink) Header() http.Header { return w.header }
func (w *responseSink) WriteHeader(status int) {
	w.status = status
}
func (w *responseSink) Write(value []byte) (int, error) {
	if w.status == 0 {
		w.status = http.StatusOK
	}
	return len(value), nil
}

func measureHTTP(enabled bool, operations int) measurement {
	responseBody := []byte(`{"items":[{"runId":"run_benchmark","state":"running"}],"page":{"hasMore":false}}`)
	handler := http.Handler(http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		if request.URL.Path != "/v1/runs" || request.Header.Get("Authorization") != "Bearer benchmark-token" {
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(responseBody)
	}))
	if enabled {
		collector := contractorperformance.New(contractorperformance.Options{})
		handler = collector.Wrap(contractorperformance.Public, handler)
	}
	request := httptest.NewRequest(http.MethodGet, "/v1/runs?limit=50", nil)
	request.Header.Set("Authorization", "Bearer benchmark-token")
	writer := &responseSink{header: make(http.Header)}
	return measureFixed(
		map[bool]string{false: "http_disabled", true: "http_enabled"}[enabled],
		operations,
		func() {
			writer.status = 0
			handler.ServeHTTP(writer, request)
			if writer.status != http.StatusOK {
				panic("representative HTTP request failed")
			}
		},
	)
}

type benchmarkClock struct{ at time.Time }

func (c *benchmarkClock) Now() time.Time { return c.at }
func (*benchmarkClock) NewTimer(time.Duration) contractorperformance.Timer {
	panic("benchmark clock timer is not used")
}

type diagnosticBackend struct {
	reads  atomic.Uint64
	sizes  atomic.Uint64
	writes atomic.Uint64
}

func (b *diagnosticBackend) ReadDatabase(context.Context) (contractorperformance.Database, contractorperformance.Reason) {
	count := b.reads.Add(1)
	return contractorperformance.Database{
		Commits: ptr(count * 10), Rollbacks: ptr(uint64(0)), Deadlocks: ptr(uint64(0)),
		TempFiles: ptr(uint64(0)), TempBytes: ptr(uint64(0)), BlocksRead: ptr(count), BlocksHit: ptr(count * 9),
	}, ""
}
func (b *diagnosticBackend) ReadSize(context.Context) (contractorperformance.DatabaseSize, contractorperformance.Reason) {
	b.sizes.Add(1)
	return contractorperformance.DatabaseSize{SizeBytes: ptr(uint64(64 << 20))}, ""
}
func (b *diagnosticBackend) Flush(context.Context, []contractorperformance.Minute, time.Time) error {
	b.writes.Add(1)
	return nil
}
func (*diagnosticBackend) Close() {}

func measureCollection(enabled bool, operations int) measurement {
	clock := &benchmarkClock{at: time.Date(2026, 9, 6, 0, 0, 0, 0, time.UTC)}
	if !enabled {
		var sink atomic.Uint64
		return measureFixed("collection_disabled", operations, func() {
			clock.at = clock.at.Add(contractorperformance.SampleInterval)
			sink.Add(1)
		})
	}
	backend := &diagnosticBackend{}
	diagnostics := contractorperformance.NewDiagnostics(contractorperformance.DiagnosticOptions{
		Clock: clock,
		Open: func(context.Context) (contractorperformance.DiagnosticBackend, error) {
			return backend, nil
		},
	})
	var count uint64
	collector := contractorperformance.New(contractorperformance.Options{
		Clock: clock,
		ReadProcess: func() (contractorperformance.Process, contractorperformance.Reason) {
			count++
			return contractorperformance.Process{
				CPUUserSeconds: ptr(float64(count) / 10), CPUSystemSeconds: ptr(float64(count) / 20),
				RSSBytes: ptr(uint64(128<<20) + count), HeapLiveBytes: ptr(uint64(32 << 20)),
			}, ""
		},
		ReadPool: func() (contractorperformance.Pool, contractorperformance.Reason) {
			return contractorperformance.Pool{
				AcquiredConnections: ptr(uint32(2)), IdleConnections: ptr(uint32(2)),
				TotalConnections: ptr(uint32(4)), MaxConnections: ptr(uint32(20)), AcquireCount: ptr(count),
			}, ""
		},
		Diagnostics: diagnostics,
	})
	index := 0
	result := measureFixed("collection_enabled", operations, func() {
		index++
		clock.at = clock.at.Add(contractorperformance.SampleInterval)
		collector.Collect()
		if index%4 == 0 {
			diagnostics.Cycle(context.Background())
		}
	})
	view := collector.Snapshot()
	diagnosticView := diagnostics.Snapshot()
	result.RetainedFrames = view.RetainedFrames
	result.RetainedBytes = view.RetainedBytes
	result.PendingMinutes = diagnosticView.PendingMinutes
	result.DatabaseReads = backend.reads.Load()
	result.DatabaseSizeReads = backend.sizes.Load()
	result.HistoryWrites = backend.writes.Load()
	return result
}

func measureFixed(name string, operations int, operation func()) measurement {
	runtime.GC()
	beforeCPU := processCPUSeconds()
	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)
	latencies := make([]int64, operations)
	started := time.Now()
	for index := range operations {
		at := time.Now()
		operation()
		latencies[index] = time.Since(at).Nanoseconds()
	}
	wall := time.Since(started)
	runtime.ReadMemStats(&after)
	return measurement{
		Scenario:            name,
		Operations:          operations,
		WallSeconds:         wall.Seconds(),
		CPUSeconds:          math.Max(0, processCPUSeconds()-beforeCPU),
		RSSBytes:            residentHighWaterBytes(),
		AllocationsPerOp:    float64(after.Mallocs-before.Mallocs) / float64(operations),
		ThroughputPerSecond: float64(operations) / wall.Seconds(),
		P95Nanoseconds:      percentile95(latencies),
	}
}

var benchmarkDigest [32]byte

func measureProfiling(name string, duration time.Duration) (measurement, error) {
	var cancel context.CancelFunc
	var serverDone chan error
	var profileDone chan profileResponse
	if name != "pprof_off" {
		listener, err := net.Listen("tcp", "127.0.0.1:0")
		if err != nil {
			return measurement{}, err
		}
		server, err := contractorprofiling.NewWithListener(listener, 2*time.Second)
		if err != nil {
			_ = listener.Close()
			return measurement{}, err
		}
		ctx, stop := context.WithCancel(context.Background())
		cancel = stop
		serverDone = make(chan error, 1)
		go func() { serverDone <- server.Run(ctx) }()
		client := &http.Client{Timeout: duration + 3*time.Second}
		response, err := client.Get("http://" + server.Addr().String() + "/debug/pprof/")
		if err != nil {
			cancel()
			<-serverDone
			return measurement{}, err
		}
		_ = response.Body.Close()
		if response.StatusCode != http.StatusOK {
			cancel()
			<-serverDone
			return measurement{}, fmt.Errorf("profiling readiness status %d", response.StatusCode)
		}
		if name == "pprof_active" {
			profileDone = make(chan profileResponse, 1)
			go func() {
				response, requestErr := client.Get("http://" + server.Addr().String() + "/debug/pprof/profile?seconds=" + strconv.Itoa(int(duration/time.Second)))
				if requestErr != nil {
					profileDone <- profileResponse{err: requestErr}
					return
				}
				defer response.Body.Close()
				bytes, readErr := io.Copy(io.Discard, response.Body)
				profileDone <- profileResponse{status: response.StatusCode, bytes: uint64(bytes), err: readErr}
			}()
			time.Sleep(25 * time.Millisecond)
		}
	}

	runtime.GC()
	beforeCPU := processCPUSeconds()
	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)
	deadline := time.Now().Add(duration)
	latencies := make([]int64, 0, 32_768)
	operations := 0
	payload := []byte("contractor bounded profiling comparison workload")
	started := time.Now()
	for time.Now().Before(deadline) {
		at := time.Now()
		for range 128 {
			benchmarkDigest = sha256.Sum256(payload)
			operations++
		}
		latencies = append(latencies, time.Since(at).Nanoseconds()/128)
	}
	wall := time.Since(started)
	runtime.ReadMemStats(&after)
	result := measurement{
		Scenario:            name,
		Operations:          operations,
		WallSeconds:         wall.Seconds(),
		CPUSeconds:          math.Max(0, processCPUSeconds()-beforeCPU),
		RSSBytes:            residentHighWaterBytes(),
		AllocationsPerOp:    float64(after.Mallocs-before.Mallocs) / float64(operations),
		ThroughputPerSecond: float64(operations) / wall.Seconds(),
		P95Nanoseconds:      percentile95(latencies),
	}
	if profileDone != nil {
		profile := <-profileDone
		if profile.err != nil || profile.status != http.StatusOK || profile.bytes == 0 {
			cancel()
			<-serverDone
			return measurement{}, fmt.Errorf("active profile result = %+v", profile)
		}
		result.ProfileResponseBytes = profile.bytes
	}
	if cancel != nil {
		cancel()
		if err := <-serverDone; err != nil {
			return measurement{}, err
		}
	}
	return result, nil
}

type profileResponse struct {
	status int
	bytes  uint64
	err    error
}

func processCPUSeconds() float64 {
	var usage unix.Rusage
	if err := unix.Getrusage(unix.RUSAGE_SELF, &usage); err != nil {
		return 0
	}
	return float64(unix.TimevalToNsec(usage.Utime)+unix.TimevalToNsec(usage.Stime)) / float64(time.Second)
}

func residentHighWaterBytes() uint64 {
	file, err := os.Open("/proc/self/status")
	if err != nil {
		var memory runtime.MemStats
		runtime.ReadMemStats(&memory)
		return memory.Sys
	}
	defer file.Close()
	values := map[string]uint64{}
	scanner := bufio.NewScanner(io.LimitReader(file, 1<<20))
	for scanner.Scan() {
		fields := strings.Fields(scanner.Text())
		if len(fields) == 3 && (fields[0] == "VmHWM:" || fields[0] == "VmRSS:") && fields[2] == "kB" {
			value, parseErr := strconv.ParseUint(fields[1], 10, 64)
			if parseErr == nil {
				values[fields[0]] = value * 1024
			}
		}
	}
	if values["VmHWM:"] != 0 {
		return values["VmHWM:"]
	}
	if values["VmRSS:"] != 0 {
		return values["VmRSS:"]
	}
	var memory runtime.MemStats
	runtime.ReadMemStats(&memory)
	return memory.Sys
}

func percentile95(values []int64) float64 {
	if len(values) == 0 {
		return 0
	}
	sort.Slice(values, func(left, right int) bool { return values[left] < values[right] })
	index := int(math.Ceil(float64(len(values))*.95)) - 1
	return float64(values[max(0, index)])
}

func summarize(samples []measurement) summary {
	throughput := make([]float64, len(samples))
	p95 := make([]float64, len(samples))
	cpu := make([]float64, len(samples))
	rss := make([]float64, len(samples))
	allocations := make([]float64, len(samples))
	for index, sample := range samples {
		throughput[index] = sample.ThroughputPerSecond
		p95[index] = sample.P95Nanoseconds
		cpu[index] = sample.CPUSeconds
		rss[index] = float64(sample.RSSBytes)
		allocations[index] = sample.AllocationsPerOp
	}
	throughputMean, throughputCV := meanCV(throughput)
	p95Mean, p95CV := meanCV(p95)
	cpuMean, _ := meanCV(cpu)
	rssMean, _ := meanCV(rss)
	allocationMean, _ := meanCV(allocations)
	return summary{
		ThroughputMean: throughputMean, ThroughputCV: throughputCV,
		P95Mean: p95Mean, P95CV: p95CV, CPUMean: cpuMean,
		RSSMean: rssMean, AllocationsMean: allocationMean,
	}
}

func meanCV(values []float64) (float64, float64) {
	if len(values) == 0 {
		return 0, 0
	}
	var total float64
	for _, value := range values {
		total += value
	}
	mean := total / float64(len(values))
	if mean == 0 {
		return mean, 0
	}
	var variance float64
	for _, value := range values {
		difference := value - mean
		variance += difference * difference
	}
	return mean, math.Sqrt(variance/float64(len(values))) / mean
}

func ptr[T any](value T) *T { return &value }

func fatal(err error) {
	_, _ = fmt.Fprintln(os.Stderr, err)
	os.Exit(1)
}
