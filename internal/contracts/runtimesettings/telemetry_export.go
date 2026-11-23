package runtimesettings

import (
	"github.com/grauwolf32/contractor/internal/contracts"
)

// TelemetryExportSettings bounds one Worker's allocation-local OTLP exporter.
type TelemetryExportSettings struct {
	BatchSizeBytes  int                     `json:"batchSizeBytes" strictjson:"required"`
	MaxAttempts     int                     `json:"maxAttempts" strictjson:"required"`
	MaxPendingSpans int                     `json:"maxPendingSpans" strictjson:"required"`
	MaxPendingBytes int                     `json:"maxPendingBytes" strictjson:"required"`
	Retry           *TelemetryRetrySettings `json:"retry,omitempty"`
}

// TelemetryExport is the export block of an allocation's RuntimeSettings. The
// Server pins every value (RCF-R102), so unlike a RuntimeConfig's
// TelemetryExportSettings, whose retry object may be omitted, it always
// carries one; the Runtime has no defaults of its own.
type TelemetryExport struct {
	BatchSizeBytes  int                    `json:"batchSizeBytes" strictjson:"required"`
	MaxAttempts     int                    `json:"maxAttempts" strictjson:"required"`
	MaxPendingSpans int                    `json:"maxPendingSpans" strictjson:"required"`
	MaxPendingBytes int                    `json:"maxPendingBytes" strictjson:"required"`
	Retry           TelemetryRetrySettings `json:"retry" strictjson:"required"`
}

// PinTelemetryExport returns the export block the Server sends for a Worker
// RuntimeConfig's export settings: the configured values, or the defaults of
// RCF-R99 and RCF-R103 for an omitted export or retry block. The result is a
// copy that never aliases configured.
func PinTelemetryExport(configured *TelemetryExportSettings) TelemetryExport {
	settings := DefaultTelemetryExportSettings()
	if configured != nil {
		settings = *configured
	}
	retry := DefaultTelemetryRetrySettings()
	if settings.Retry != nil {
		retry = *settings.Retry
	}
	return TelemetryExport{
		BatchSizeBytes: settings.BatchSizeBytes, MaxAttempts: settings.MaxAttempts,
		MaxPendingSpans: settings.MaxPendingSpans, MaxPendingBytes: settings.MaxPendingBytes,
		Retry: retry,
	}
}

// Validate applies the bounds TelemetryExportSettings defines for both forms.
func (e TelemetryExport) Validate() error {
	retry := e.Retry
	return TelemetryExportSettings{
		BatchSizeBytes: e.BatchSizeBytes, MaxAttempts: e.MaxAttempts,
		MaxPendingSpans: e.MaxPendingSpans, MaxPendingBytes: e.MaxPendingBytes,
		Retry: &retry,
	}.Validate()
}

func DefaultTelemetryExportSettings() TelemetryExportSettings {
	return TelemetryExportSettings{
		BatchSizeBytes: 8 * 1024 * 1024, MaxAttempts: 2,
		MaxPendingSpans: 2048, MaxPendingBytes: 64 * 1024 * 1024,
	}
}

func (s TelemetryExportSettings) Validate() error {
	if s.Retry != nil {
		if err := s.Retry.Validate(); err != nil {
			return err
		}
	}
	if s.BatchSizeBytes < 1024*1024 || s.BatchSizeBytes > 64*1024*1024 {
		return contracts.Invalidf("telemetry export batchSizeBytes must be from 1 through 64 MiB")
	}
	if s.MaxAttempts < 1 || s.MaxAttempts > 10 {
		return contracts.Invalidf("telemetry export maxAttempts must be from 1 through 10")
	}
	if s.MaxPendingSpans < 1 || s.MaxPendingSpans > 2048 {
		return contracts.Invalidf("telemetry export maxPendingSpans must be from 1 through 2048")
	}
	if s.MaxPendingBytes < s.BatchSizeBytes || s.MaxPendingBytes > 64*1024*1024 {
		return contracts.Invalidf("telemetry export maxPendingBytes must cover batchSizeBytes and be at most 64 MiB")
	}
	return nil
}

// TelemetryRetrySettings is optional so stored RuntimeConfig digests remain stable.
type TelemetryRetrySettings struct {
	InitialBackoffMilliseconds int `json:"initialBackoffMilliseconds" strictjson:"required"`
	MaxBackoffMilliseconds     int `json:"maxBackoffMilliseconds" strictjson:"required"`
}

func DefaultTelemetryRetrySettings() TelemetryRetrySettings {
	return TelemetryRetrySettings{InitialBackoffMilliseconds: 100, MaxBackoffMilliseconds: 1000}
}

func (s TelemetryRetrySettings) Validate() error {
	if s.InitialBackoffMilliseconds < 1 || s.MaxBackoffMilliseconds > 60000 || s.InitialBackoffMilliseconds > s.MaxBackoffMilliseconds {
		return contracts.Invalidf("telemetry retry backoff must satisfy 1 <= initialBackoffMilliseconds <= maxBackoffMilliseconds <= 60000")
	}
	return nil
}
