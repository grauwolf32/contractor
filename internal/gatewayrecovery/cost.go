package gatewayrecovery

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"

	"github.com/grauwolf32/contractor/internal/contracts/reporting"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
)

type costCollectorKey struct{}
type costPhaseKey struct{}
type costModelKey struct{}

// WithLLMCostCollector binds the existing Planner invocation's cost owner.
func WithLLMCostCollector(ctx context.Context, collector *reporting.LLMCostCollector) context.Context {
	return context.WithValue(ctx, costCollectorKey{}, collector)
}

// WithLLMCostPhase labels a tool-free auxiliary call without changing its route.
func WithLLMCostPhase(ctx context.Context, phase string) context.Context {
	return context.WithValue(ctx, costPhaseKey{}, phase)
}

// WithLLMCostModel uses the selected alias, never an untrusted response label.
func WithLLMCostModel(ctx context.Context, model string) context.Context {
	return context.WithValue(ctx, costModelKey{}, model)
}

func observeLLMCost(ctx context.Context, headers http.Header, body []byte) {
	collector, _ := ctx.Value(costCollectorKey{}).(*reporting.LLMCostCollector)
	if collector == nil {
		return
	}
	phase, _ := ctx.Value(costPhaseKey{}).(string)
	if phase == "" {
		phase = "main"
	}
	model, _ := ctx.Value(costModelKey{}).(string)
	source, amount := responseCost(model, headers, body)
	collector.Observe(model, phase, source, amount)
}

// responseCost adapts only documented USD observations. Provider body/header
// observations must agree; Gateway totals are a separate basis, not an addend.
func responseCost(model string, headers http.Header, body []byte) (string, string) {
	provider := headers.Get("llm_provider-x-litellm-response-cost")
	var doc map[string]json.RawMessage
	if strings.HasPrefix(model, "openrouter/") && len(body) > 0 && strictjson.Decode(body, &doc) == nil {
		var usage map[string]json.RawMessage
		if strictjson.Decode(doc["usage"], &usage) == nil {
			// BYOK has separate router and upstream billing scopes. It is not a
			// complete inference charge in this initial USD observation contract.
			if string(usage["is_byok"]) == "true" {
				return "", ""
			}
			if raw, exists := usage["cost"]; exists && string(raw) != "null" {
				amount, ok := reporting.NormalizeUSD(string(raw))
				if !ok {
					return "", ""
				}
				if provider != "" {
					other, valid := reporting.NormalizeUSD(provider)
					if !valid || other != amount {
						return "", ""
					}
				}
				return "provider", amount
			}
		}
	}
	if provider != "" {
		if amount, ok := reporting.NormalizeUSD(provider); ok {
			return "provider", amount
		}
		return "", ""
	}
	if amount, ok := reporting.NormalizeUSD(headers.Get("x-litellm-response-cost")); ok {
		return "gateway", amount
	}
	return "", ""
}
