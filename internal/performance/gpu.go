package performance

import (
	"bytes"
	"context"
	"encoding/csv"
	"errors"
	"io"
	"math"
	"os"
	"os/exec"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"
	"unicode"
)

const MaxGPUDevices = 8
const gpuProbeTimeout = 2 * time.Second
const gpuProbeBytes = 16 * 1024

// GPU measurements cover the entire physical device visible to the Server,
// including other applications. They do not describe a Run or Runtime process.
type GPU struct {
	Freshness Freshness   `json:"freshness"`
	Devices   []GPUDevice `json:"devices"`
}

type GPUDevice struct {
	ID                 string   `json:"id"`
	Name               string   `json:"name"`
	UtilizationPercent *float64 `json:"utilizationPercent,omitempty"`
	MemoryUsedBytes    *uint64  `json:"memoryUsedBytes,omitempty"`
	MemoryTotalBytes   *uint64  `json:"memoryTotalBytes,omitempty"`
	TemperatureCelsius *float64 `json:"temperatureCelsius,omitempty"`
	PowerWatts         *float64 `json:"powerWatts,omitempty"`
	PowerLimitWatts    *float64 `json:"powerLimitWatts,omitempty"`
}

type GPUReader func(context.Context) ([]GPUDevice, Reason)

// Construction starts no process and returns nil if nvidia-smi is not on PATH.
// Fixed arguments, no shell, no process list, bounded output/deadline and
// discarded stderr keep this an optional probe, never a Server prerequisite.
func NewNVIDIAGPUReader() GPUReader {
	binary, err := exec.LookPath("nvidia-smi")
	if err != nil {
		return nil
	}
	return func(ctx context.Context) ([]GPUDevice, Reason) {
		ctx, cancel := context.WithTimeout(ctx, gpuProbeTimeout)
		defer cancel()
		cmd := exec.CommandContext(ctx, binary, "--query-gpu=uuid,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit", "--format=csv,noheader,nounits")
		cmd.Env = append(cmd.Environ(), "LC_ALL=C")
		cmd.WaitDelay = 100 * time.Millisecond
		output := &gpuOutput{}
		cmd.Stdout, cmd.Stderr = output, io.Discard
		if err := cmd.Run(); err != nil {
			switch {
			case ctx.Err() != nil:
				return nil, BudgetExceeded
			case errors.Is(err, os.ErrNotExist):
				return nil, GPUNotAvailable
			case output.exceeded:
				return nil, RecordLimit
			default:
				return nil, ReadFailed
			}
		}
		return parseNVIDIAGPUs(output.buffer.Bytes())
	}
}

type gpuOutput struct {
	buffer   bytes.Buffer
	exceeded bool
}

func (b *gpuOutput) Write(p []byte) (int, error) {
	if len(p) > gpuProbeBytes-b.buffer.Len() {
		b.exceeded = true
		return 0, errors.New("GPU probe output exceeded bound")
	}
	return b.buffer.Write(p)
}

var gpuIDPattern = regexp.MustCompile(`^GPU-[0-9a-fA-F-]{1,64}$`)

func validGPUIdentity(id, name string) bool {
	return gpuIDPattern.MatchString(id) && strings.TrimSpace(name) != "" && len(name) <= 128 && strings.IndexFunc(name, unicode.IsControl) < 0
}

func parseNVIDIAGPUs(raw []byte) ([]GPUDevice, Reason) {
	if len(raw) > gpuProbeBytes {
		return nil, RecordLimit
	}
	reader := csv.NewReader(bytes.NewReader(raw))
	reader.TrimLeadingSpace, reader.FieldsPerRecord = true, 8
	devices := []GPUDevice{}
	reason := Reason("")
	seen := map[string]bool{}
	for {
		row, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, ReadFailed
		}
		if len(devices) == MaxGPUDevices {
			reason = RecordLimit
			break
		}
		d := GPUDevice{ID: strings.TrimSpace(row[0]), Name: strings.TrimSpace(row[1])}
		if !validGPUIdentity(d.ID, d.Name) || seen[d.ID] {
			return nil, ReadFailed
		}
		seen[d.ID] = true
		values := make([]*float64, 6)
		for i, field := range row[2:] {
			field = strings.TrimSpace(field)
			if field == "N/A" || field == "[N/A]" || field == "[Not Supported]" {
				if reason == "" {
					reason = UnsupportedMetric
				}
				continue
			}
			n, err := strconv.ParseFloat(field, 64)
			if err != nil || n < 0 || math.IsInf(n, 0) || math.IsNaN(n) {
				return nil, ReadFailed
			}
			values[i] = &n
		}
		d.UtilizationPercent, d.TemperatureCelsius, d.PowerWatts, d.PowerLimitWatts = values[0], values[3], values[4], values[5]
		for i, target := range []**uint64{&d.MemoryUsedBytes, &d.MemoryTotalBytes} {
			if value := values[i+1]; value != nil {
				// nvidia-smi reports framebuffer memory in MiB, not decimal MB.
				if *value > float64(1<<53-1)/(1024*1024) {
					return nil, ReadFailed
				}
				n := uint64(*value * 1024 * 1024)
				*target = &n
			}
		}
		if !validGPUDevice(d) {
			return nil, ReadFailed
		}
		devices = append(devices, d)
	}
	if len(devices) == 0 {
		return devices, GPUNotAvailable
	}
	sort.Slice(devices, func(i, j int) bool { return devices[i].ID < devices[j].ID })
	return devices, reason
}

func validGPUDevice(d GPUDevice) bool {
	if !validGPUIdentity(d.ID, d.Name) {
		return false
	}
	for _, v := range []*float64{d.UtilizationPercent, d.TemperatureCelsius, d.PowerWatts, d.PowerLimitWatts} {
		if v != nil && (*v < 0 || math.IsNaN(*v) || math.IsInf(*v, 0)) {
			return false
		}
	}
	for _, v := range []*uint64{d.MemoryUsedBytes, d.MemoryTotalBytes} {
		if v != nil && *v > 1<<53-1 {
			return false
		}
	}
	return (d.UtilizationPercent == nil || *d.UtilizationPercent <= 100) &&
		(d.MemoryUsedBytes == nil || d.MemoryTotalBytes == nil || *d.MemoryUsedBytes <= *d.MemoryTotalBytes)
}

func (g GPU) Validate() error {
	if g.Freshness.Validate() != nil || g.Freshness.IntervalSeconds != 15 || g.Devices == nil || len(g.Devices) > MaxGPUDevices {
		return errInvalidRecord
	}
	if (g.Freshness.Status == Unavailable) != (len(g.Devices) == 0) {
		return errInvalidRecord
	}
	seen := map[string]bool{}
	for _, d := range g.Devices {
		if seen[d.ID] || !validGPUDevice(d) {
			return errInvalidRecord
		}
		seen[d.ID] = true
	}
	return nil
}
