// Package gatewayrecovery coordinates model availability across execution lanes.
package gatewayrecovery

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"time"
)

type Policy struct {
	RequestTimeout  time.Duration
	InitialDelay    time.Duration
	MaxDelay        time.Duration
	AutomaticWindow time.Duration
}

// Defaults allow a short local model reload, while keeping probes infrequent.
// Operators can change every duration through ServerConfig.llmRecovery.
func DefaultPolicy() Policy {
	return Policy{
		RequestTimeout:  time.Minute,
		InitialDelay:    time.Second,
		MaxDelay:        30 * time.Second,
		AutomaticWindow: 5 * time.Minute,
	}
}
func (p Policy) Validate() error {
	if p.RequestTimeout < time.Second || p.InitialDelay < time.Second || p.MaxDelay < p.InitialDelay || p.AutomaticWindow < p.MaxDelay {
		return errors.New("LLM recovery requires positive request timeout and 1s <= initialDelay <= maxDelay <= automaticWindow")
	}
	return nil
}

type Route struct {
	OwnerID         string
	GatewayDigest   string
	Model           string
	CredentialID    string
	TransportDigest string
}

func (r Route) Key() string {
	encoded, _ := json.Marshal(r)
	sum := sha256.Sum256(encoded)
	return hex.EncodeToString(sum[:])
}

type Decision struct {
	Allowed               bool    `json:"allowed"`
	Code                  string  `json:"code,omitempty"`
	RetryAfterSeconds     float64 `json:"retryAfterSeconds"`
	RequestTimeoutSeconds float64 `json:"requestTimeoutSeconds"`
	RequiresRetry         bool    `json:"requiresRetry"`
}

var ErrUnavailable = errors.New("gateway recovery authority is unavailable")
var ErrInvalid = errors.New("invalid gateway recovery request")

func ValidFailure(code string) bool {
	switch code {
	case "model_unavailable", "gateway_unavailable", "gateway_timeout", "gateway_rate_limited":
		return true
	default:
		return false
	}
}
