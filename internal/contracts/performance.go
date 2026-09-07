package contracts

import (
	"bytes"
	"encoding/json"
	"math"
)

const PerformanceMetricsVersion = 1
const PerformanceMetricsIntervalSeconds = 15
const MaxResourceInteger = 1<<53 - 1 // exact in all JSON consumers, including TypeScript

// PerformanceCollectionPolicy is the Server decision pinned with an
// allocation. Legacy is a read projection for rows created before the policy
// column existed and must never be written for a new allocation.
type PerformanceCollectionPolicy string

const (
	PerformanceCollectionRequested   PerformanceCollectionPolicy = "requested"
	PerformanceCollectionDisabled    PerformanceCollectionPolicy = "disabled"
	PerformanceCollectionUnsupported PerformanceCollectionPolicy = "unsupported"
	PerformanceCollectionLegacy      PerformanceCollectionPolicy = "legacy"
)

func (p PerformanceCollectionPolicy) ValidatePinned() error {
	switch p {
	case PerformanceCollectionRequested, PerformanceCollectionDisabled, PerformanceCollectionUnsupported:
		return nil
	default:
		return invalidf("invalid performance collection policy")
	}
}

func (p PerformanceCollectionPolicy) Request() *PerformanceMetricsRequest {
	if p != PerformanceCollectionRequested {
		return nil
	}
	return &PerformanceMetricsRequest{
		Version: PerformanceMetricsVersion, IntervalSeconds: PerformanceMetricsIntervalSeconds,
	}
}

type PerformanceMetricsVersions []int

func (v *PerformanceMetricsVersions) UnmarshalJSON(data []byte) error {
	var versions []int
	if err := json.Unmarshal(data, &versions); err != nil || versions == nil {
		return invalidf("invalid performance metrics capability")
	}
	*v = versions
	return nil
}

type PerformanceMetricsRequest struct {
	Version         int `json:"version"`
	IntervalSeconds int `json:"intervalSeconds"`
}

func (r PerformanceMetricsRequest) Validate() error {
	if r.Version != PerformanceMetricsVersion || r.IntervalSeconds != PerformanceMetricsIntervalSeconds {
		return invalidf("unsupported performance metrics request")
	}
	return nil
}

type ResourceStatus string
type ResourceReason string

const (
	ResourceComplete            ResourceStatus = "complete"
	ResourcePartial             ResourceStatus = "partial"
	ResourceUnavailable         ResourceStatus = "unavailable"
	ResourceUnsupportedPlatform ResourceReason = "unsupported_platform"
	ResourceReadFailed          ResourceReason = "read_failed"
	ResourceSamplingGap         ResourceReason = "sampling_gap"
	ResourceCounterReset        ResourceReason = "counter_reset"
	ResourceInvalidReport       ResourceReason = "invalid_report"
)

// RuntimeResources describes allocation-local observations of the entire Runtime
// process. Pointers distinguish unknown measurements from actual zero values.
type RuntimeResources struct {
	Version              int             `json:"version"`
	Scope                string          `json:"scope"`
	Status               ResourceStatus  `json:"status"`
	Reason               *ResourceReason `json:"reason,omitempty"`
	DurationSeconds      *float64        `json:"durationSeconds,omitempty"`
	CPUUserSeconds       *float64        `json:"cpuUserSeconds,omitempty"`
	CPUSystemSeconds     *float64        `json:"cpuSystemSeconds,omitempty"`
	RSSStartBytes        *uint64         `json:"rssStartBytes,omitempty"`
	RSSEndBytes          *uint64         `json:"rssEndBytes,omitempty"`
	RSSPeakObservedBytes *uint64         `json:"rssPeakObservedBytes,omitempty"`
	RSSSampleCount       *uint64         `json:"rssSampleCount,omitempty"`
	MaxSampleGapSeconds  *float64        `json:"maxSampleGapSeconds,omitempty"`
}

func (r *RuntimeResources) UnmarshalJSON(data []byte) error {
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(data, &fields); err != nil {
		return invalidf("invalid resources")
	}
	if fields == nil {
		return invalidf("resources must be an object")
	}
	for _, raw := range fields {
		if bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
			return invalidf("unknown resource fields must be omitted, not null")
		}
	}
	type wireResources RuntimeResources
	var value wireResources
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&value); err != nil {
		return invalidf("invalid resource fields")
	}
	*r = RuntimeResources(value)
	return nil
}

func (r RuntimeResources) Validate() error {
	if r.Version != 1 || r.Scope != "runtime_process" {
		return invalidf("invalid resource version or scope")
	}
	switch r.Status {
	case ResourceComplete, ResourcePartial, ResourceUnavailable:
	default:
		return invalidf("invalid resource status")
	}
	if r.Reason != nil {
		switch *r.Reason {
		case ResourceUnsupportedPlatform, ResourceReadFailed, ResourceSamplingGap, ResourceCounterReset, ResourceInvalidReport:
		default:
			return invalidf("invalid resource reason")
		}
	}
	for _, value := range []*float64{r.DurationSeconds, r.CPUUserSeconds, r.CPUSystemSeconds, r.MaxSampleGapSeconds} {
		if value != nil && (*value < 0 || math.IsNaN(*value) || math.IsInf(*value, 0)) {
			return invalidf("invalid resource measurement")
		}
	}
	for _, value := range []*uint64{r.RSSStartBytes, r.RSSEndBytes, r.RSSPeakObservedBytes, r.RSSSampleCount} {
		if value != nil && *value > MaxResourceInteger {
			return invalidf("resource integer exceeds exact JSON range")
		}
	}
	if r.MaxSampleGapSeconds != nil && r.DurationSeconds != nil && *r.MaxSampleGapSeconds > *r.DurationSeconds {
		return invalidf("resource sampling gap exceeds duration")
	}
	hasSamples := r.RSSSampleCount != nil && *r.RSSSampleCount > 0
	if hasSamples != (r.RSSPeakObservedBytes != nil) {
		return invalidf("resource peak and sample count are inconsistent")
	}
	boundaries := uint64(0)
	for _, value := range []*uint64{r.RSSStartBytes, r.RSSEndBytes} {
		if value != nil {
			boundaries++
			if !hasSamples || *value > *r.RSSPeakObservedBytes {
				return invalidf("resource boundary exceeds observed peak")
			}
		}
	}
	if hasSamples && *r.RSSSampleCount < boundaries {
		return invalidf("resource sample count omits boundaries")
	}
	if r.Status == ResourceComplete && (r.Reason != nil || r.DurationSeconds == nil || r.CPUUserSeconds == nil || r.CPUSystemSeconds == nil || boundaries != 2 || r.MaxSampleGapSeconds == nil || *r.MaxSampleGapSeconds > 30) {
		return invalidf("complete resources require successful boundaries and bounded coverage")
	}
	return nil
}

// Resource-only errors are isolated from execution truth. Duplicate keys and
// malformed JSON remain envelope errors at the strict transport decoder.
func decodeOptionalResources(data json.RawMessage) (*RuntimeResources, *ResourceReason) {
	if len(data) == 0 {
		return nil, nil
	}
	resources, err := DecodePrivateStrict[RuntimeResources](data)
	if err != nil {
		reason := ResourceInvalidReport
		return nil, &reason
	}
	return &resources, nil
}
