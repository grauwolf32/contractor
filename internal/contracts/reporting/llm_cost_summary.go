package reporting

import (
	"math/big"
	"sort"
	"time"
)

type CostBreakdownRow struct {
	Key    string `json:"key"`
	Label  string `json:"label"`
	Calls  string `json:"calls"`
	Amount string `json:"amount"`
	Basis  string `json:"basis"`
}

type LLMCostExecution struct {
	ExecutionID string  `json:"executionId"`
	RunID       *string `json:"runId"`
	Amount      *string `json:"amount"`
	Basis       string  `json:"basis"`
	Coverage    string  `json:"coverage"`
	Pending     bool    `json:"pending"`
}

type LLMCostSummary struct {
	Scope            string             `json:"scope"`
	ScopeID          string             `json:"scopeId"`
	Currency         string             `json:"currency"`
	Amount           *string            `json:"amount"`
	Calls            string             `json:"calls"`
	PricedCalls      string             `json:"pricedCalls"`
	Reports          string             `json:"reports"`
	UnknownReports   string             `json:"unknownReports"`
	Basis            string             `json:"basis"`
	Coverage         string             `json:"coverage"`
	Pending          bool               `json:"pending"`
	HistoricalGap    bool               `json:"historicalGap"`
	AsOf             time.Time          `json:"asOf"`
	UpdatedAt        *time.Time         `json:"updatedAt"`
	ChecksAmount     *string            `json:"checksAmount"`
	StandaloneAmount *string            `json:"standaloneAmount"`
	Activity         []CostBreakdownRow `json:"activity"`
	Models           []CostBreakdownRow `json:"models"`
	Executions       []LLMCostExecution `json:"executions"`
}

func BuildLLMCostSummary(scope, id string, t LLMCostTotals, pending, hasWork bool, asOf time.Time) LLMCostSummary {
	value := LLMCostSummary{Scope: scope, ScopeID: id, Currency: t.Currency, Amount: t.KnownAmount(), Calls: t.Calls, PricedCalls: t.PricedCalls, Reports: t.Reports, UnknownReports: t.UnknownReports, Basis: "none", Coverage: "complete", Pending: pending, HistoricalGap: t.HistoricalGap, AsOf: asOf, UpdatedAt: t.UpdatedAt, Activity: []CostBreakdownRow{}, Models: []CostBreakdownRow{}}
	value.Executions = []LLMCostExecution{}
	if !hasWork && t.Reports == "0" && !t.HistoricalGap {
		zero := "0"
		value.Amount = &zero
	}
	if t.ProviderCalls != "0" {
		value.Basis = "reported"
	}
	if t.GatewayCalls != "0" {
		value.Basis = "estimated"
		if t.ProviderCalls != "0" {
			value.Basis = "mixed"
		}
	}
	if value.Amount == nil {
		value.Coverage = "unavailable"
	} else if t.HistoricalGap || t.UnknownReports != "0" || t.Calls != t.PricedCalls {
		value.Coverage = "partial"
	} else if pending {
		value.Coverage = "pending"
	}
	if value.Amount != nil {
		checks, standalone := t.ChecksAmount, t.StandaloneAmount
		value.ChecksAmount = &checks
		value.StandaloneAmount = &standalone
	}
	activity, models := map[string]*CostBreakdownRow{}, map[string]*CostBreakdownRow{}
	for _, g := range t.Groups {
		if g.Calls == "0" {
			continue
		}
		key, label := g.Phase, g.Phase
		switch g.Phase {
		case "main":
			key = g.Role
			label = "Worker analysis"
			if g.Role == "planner" {
				label = "Run planning"
			}
		case "compaction":
			label = "Context compaction"
		case "finalization":
			label = "Finalization"
		case "summarization":
			label = "Summary"
		}
		mergeCostBreakdown(activity, key, label, g)
		key, label = "model:"+g.Model, g.Model
		if g.Model == "" {
			key = "other"
			label = "Other models"
		}
		mergeCostBreakdown(models, key, label, g)
	}
	for _, row := range activity {
		value.Activity = append(value.Activity, *row)
	}
	for _, row := range models {
		value.Models = append(value.Models, *row)
	}
	sort.Slice(value.Activity, func(i, j int) bool { return value.Activity[i].Key < value.Activity[j].Key })
	sort.Slice(value.Models, func(i, j int) bool { return value.Models[i].Label < value.Models[j].Label })
	return value
}

func mergeCostBreakdown(rows map[string]*CostBreakdownRow, key, label string, g CostTotalGroup) {
	row := rows[key]
	if row == nil {
		row = &CostBreakdownRow{Key: key, Label: label, Calls: "0", Amount: "0"}
		rows[key] = row
	}
	n := new(big.Int).Add(intValue(row.Calls), intValue(g.Calls))
	row.Calls = n.String()
	row.Amount = formatUSD(new(big.Int).Add(costUnits(row.Amount), costUnits(g.Amount)))
	basis := "reported"
	if g.Source == "gateway" {
		basis = "estimated"
	}
	if row.Basis == "" {
		row.Basis = basis
	} else if row.Basis != basis {
		row.Basis = "mixed"
	}
}
