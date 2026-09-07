package contracts

// TelemetryExportSettings bounds one Worker's allocation-local OTLP exporter.
type TelemetryExportSettings struct {
	BatchSizeBytes  int `json:"batchSizeBytes"`
	MaxAttempts     int `json:"maxAttempts"`
	MaxPendingSpans int `json:"maxPendingSpans"`
	MaxPendingBytes int `json:"maxPendingBytes"`
}

func DefaultTelemetryExportSettings() TelemetryExportSettings {
	return TelemetryExportSettings{
		BatchSizeBytes: 8 * 1024 * 1024, MaxAttempts: 2,
		MaxPendingSpans: 2048, MaxPendingBytes: 64 * 1024 * 1024,
	}
}

func (s TelemetryExportSettings) Validate() error {
	if s.BatchSizeBytes < 1024*1024 || s.BatchSizeBytes > 64*1024*1024 {
		return invalidf("telemetry export batchSizeBytes must be from 1 through 64 MiB")
	}
	if s.MaxAttempts < 1 || s.MaxAttempts > 10 {
		return invalidf("telemetry export maxAttempts must be from 1 through 10")
	}
	if s.MaxPendingSpans < 1 || s.MaxPendingSpans > 2048 {
		return invalidf("telemetry export maxPendingSpans must be from 1 through 2048")
	}
	if s.MaxPendingBytes < s.BatchSizeBytes || s.MaxPendingBytes > 64*1024*1024 {
		return invalidf("telemetry export maxPendingBytes must cover batchSizeBytes and be at most 64 MiB")
	}
	return nil
}
