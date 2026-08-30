// Package streamline implements the bounded streamline@1 PlannerFactory and
// the shared model-backed engine delegated to by router@1. Google ADK types do
// not cross the parent planner.Factory boundary consumed by Workflow Scheduler.
package streamline

import (
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/planner"
)

const Ref = planner.StreamlineRef

const (
	defaultMaxModelCalls   = 32
	defaultMaxTokens       = 200_000
	defaultMaxWorkerCalls  = 64
	defaultMaxWallTime     = 30 * time.Minute
	defaultMaxOutputTokens = 8_192
	defaultGatewayTimeout  = 5 * time.Minute
)

type Limits struct {
	MaxModelCalls  int
	MaxTokens      int64
	MaxWorkerCalls int
	MaxWallTime    time.Duration
}

func DefaultLimits() Limits {
	return Limits{
		MaxModelCalls: defaultMaxModelCalls, MaxTokens: defaultMaxTokens,
		MaxWorkerCalls: defaultMaxWorkerCalls, MaxWallTime: defaultMaxWallTime,
	}
}

func normalizeLimits(value Limits) (Limits, error) {
	defaults := DefaultLimits()
	if value.MaxModelCalls == 0 {
		value.MaxModelCalls = defaults.MaxModelCalls
	}
	if value.MaxTokens == 0 {
		value.MaxTokens = defaults.MaxTokens
	}
	if value.MaxWorkerCalls == 0 {
		value.MaxWorkerCalls = defaults.MaxWorkerCalls
	}
	if value.MaxWallTime == 0 {
		value.MaxWallTime = defaults.MaxWallTime
	}
	if value.MaxModelCalls <= 0 || value.MaxModelCalls > 1_000 ||
		value.MaxTokens <= 0 || value.MaxTokens > contracts.MaxWorkerTotalTokens ||
		value.MaxWorkerCalls <= 0 || value.MaxWorkerCalls > contracts.MaxPlannerWorkerCalls ||
		value.MaxWallTime <= 0 || value.MaxWallTime > 24*time.Hour {
		return Limits{}, fmt.Errorf("model-backed Planner limits are outside bounded ranges")
	}
	return value, nil
}

type GatewaySettings struct {
	URL             string
	Token           contracts.SecretString
	Model           string
	MaxOutputTokens int
	RequestTimeout  time.Duration
	HTTPClient      *http.Client
}

func normalizeGatewaySettings(settings GatewaySettings) (GatewaySettings, error) {
	parsed, err := url.Parse(strings.TrimSpace(settings.URL))
	if err != nil || parsed.Host == "" || parsed.User != nil ||
		(parsed.Scheme != "http" && parsed.Scheme != "https") ||
		parsed.RawQuery != "" || parsed.Fragment != "" {
		return GatewaySettings{}, fmt.Errorf("Planner LLM Gateway URL is invalid")
	}
	if strings.TrimSpace(settings.Model) == "" {
		return GatewaySettings{}, fmt.Errorf("Planner LLM Gateway model is required")
	}
	if settings.MaxOutputTokens == 0 {
		settings.MaxOutputTokens = defaultMaxOutputTokens
	}
	if settings.RequestTimeout == 0 {
		settings.RequestTimeout = defaultGatewayTimeout
	}
	if settings.MaxOutputTokens <= 0 || settings.MaxOutputTokens > 1_000_000 ||
		settings.RequestTimeout <= 0 || settings.RequestTimeout > 30*time.Minute {
		return GatewaySettings{}, fmt.Errorf("Planner LLM Gateway limits are invalid")
	}
	settings.URL = strings.TrimRight(parsed.String(), "/")
	if settings.HTTPClient == nil {
		settings.HTTPClient = &http.Client{}
	}
	client := *settings.HTTPClient
	client.Timeout = settings.RequestTimeout
	client.CheckRedirect = func(*http.Request, []*http.Request) error {
		return http.ErrUseLastResponse
	}
	settings.HTTPClient = &client
	return settings, nil
}
