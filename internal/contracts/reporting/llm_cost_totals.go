package reporting

import (
	"bytes"
	"encoding/json"
	"math/big"
	"reflect"
	"regexp"
	"sort"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
)

// CostContribution is the expected report slot in its existing allocation or
// Stage record. Empty ReportID means no report was received, with no observed
// calls or cost. It survives diagnostic retention, but has no independent life.
type CostContribution struct {
	ReportID   string    `json:"reportId" strictjson:"required"`
	ReceivedAt time.Time `json:"receivedAt" strictjson:"required"`
	Complete   bool      `json:"complete" strictjson:"required"`
	ModelCalls int64     `json:"modelCalls" strictjson:"required"`
	Cost       *LLMCost  `json:"cost" strictjson:"required,nullable"`
}

type CostTotalGroup struct {
	Model  string `json:"model" strictjson:"required"`
	Role   string `json:"role" strictjson:"required"`
	Phase  string `json:"phase" strictjson:"required"`
	Source string `json:"source" strictjson:"required"`
	Calls  string `json:"calls" strictjson:"required"`
	Amount string `json:"amount" strictjson:"required"`
}

// LLMCostTotals is compact operational state in an existing Run, Audit or
// Project row. Counters and amounts remain exact through every JSON consumer.
type LLMCostTotals struct {
	Currency         string           `json:"currency" strictjson:"required"`
	Reports          string           `json:"reports" strictjson:"required"`
	UnknownReports   string           `json:"unknownReports" strictjson:"required"`
	Calls            string           `json:"calls" strictjson:"required"`
	PricedCalls      string           `json:"pricedCalls" strictjson:"required"`
	ProviderCalls    string           `json:"providerCalls" strictjson:"required"`
	GatewayCalls     string           `json:"gatewayCalls" strictjson:"required"`
	Amount           string           `json:"amount" strictjson:"required"`
	ChecksAmount     string           `json:"checksAmount" strictjson:"required"`
	StandaloneAmount string           `json:"standaloneAmount" strictjson:"required"`
	HistoricalGap    bool             `json:"historicalGap" strictjson:"required"`
	UpdatedAt        *time.Time       `json:"updatedAt" strictjson:"required,nullable"`
	Groups           []CostTotalGroup `json:"groups" strictjson:"required"`
}

var costCounter = regexp.MustCompile(`^(0|[1-9][0-9]{0,37})$`)
var costTotalAmount = regexp.MustCompile(`^(0|[1-9][0-9]{0,37})(\.[0-9]{0,17}[1-9])?$`)

func NewLLMCostTotals(historicalGap bool) LLMCostTotals {
	return LLMCostTotals{Currency: "USD", Reports: "0", UnknownReports: "0", Calls: "0", PricedCalls: "0", ProviderCalls: "0", GatewayCalls: "0", Amount: "0", ChecksAmount: "0", StandaloneAmount: "0", HistoricalGap: historicalGap, Groups: []CostTotalGroup{}}
}

func DecodeLLMCostTotals(raw []byte) (LLMCostTotals, error) {
	if len(raw) == 0 || bytes.Equal(raw, []byte("null")) {
		return NewLLMCostTotals(true), nil
	}
	// Empty object is the storage encoding of a new, known-empty aggregate;
	// SQL NULL above is deliberately different: its history is unknown.
	if bytes.Equal(raw, []byte("{}")) {
		return NewLLMCostTotals(false), nil
	}
	var value LLMCostTotals
	if err := strictjson.Decode(raw, &value); err != nil {
		return value, err
	}
	return value, value.Validate()
}

func (t LLMCostTotals) Validate() error {
	if t.Currency != "USD" || t.Groups == nil || len(t.Groups) > 64 {
		return contracts.Invalidf("cost total shape is invalid")
	}
	for _, c := range []string{t.Reports, t.UnknownReports, t.Calls, t.PricedCalls, t.ProviderCalls, t.GatewayCalls} {
		if !costCounter.MatchString(c) {
			return contracts.Invalidf("cost total counter is invalid")
		}
	}
	for _, a := range []string{t.Amount, t.ChecksAmount, t.StandaloneAmount} {
		if !costTotalAmount.MatchString(a) {
			return contracts.Invalidf("cost total amount is invalid")
		}
	}
	if intValue(t.UnknownReports).Cmp(intValue(t.Reports)) > 0 || intValue(t.PricedCalls).Cmp(intValue(t.Calls)) > 0 {
		return contracts.Invalidf("cost coverage is inconsistent")
	}
	seen := map[string]bool{}
	groupCalls, providerCalls, gatewayCalls, amount := new(big.Int), new(big.Int), new(big.Int), new(big.Int)
	explicit := 0
	for _, g := range t.Groups {
		model := g.Model
		if model == "" {
			model = "other-models"
		}
		if !validCostGroup(LLMCostGroup{Model: model, Phase: g.Phase, Source: g.Source}) || (g.Role != "worker" && g.Role != "planner") || !costCounter.MatchString(g.Calls) || !costTotalAmount.MatchString(g.Amount) || seen[costGroupKey(g)] {
			return contracts.Invalidf("cost total group is invalid")
		}
		seen[costGroupKey(g)] = true
		if g.Model != "" {
			explicit++
		}
		groupCalls.Add(groupCalls, intValue(g.Calls))
		amount.Add(amount, costUnits(g.Amount))
		if g.Source == "provider" {
			providerCalls.Add(providerCalls, intValue(g.Calls))
		} else {
			gatewayCalls.Add(gatewayCalls, intValue(g.Calls))
		}
	}
	if explicit > 48 || groupCalls.Cmp(intValue(t.PricedCalls)) != 0 ||
		providerCalls.Cmp(intValue(t.ProviderCalls)) != 0 || gatewayCalls.Cmp(intValue(t.GatewayCalls)) != 0 ||
		amount.Cmp(costUnits(t.Amount)) != 0 ||
		new(big.Int).Add(costUnits(t.ChecksAmount), costUnits(t.StandaloneAmount)).Cmp(amount) != 0 {
		return contracts.Invalidf("cost total breakdown is inconsistent")
	}
	return nil
}

func DecodeCostContribution(raw []byte) (*CostContribution, error) {
	if len(raw) == 0 || bytes.Equal(raw, []byte("null")) {
		return nil, nil
	}
	var value CostContribution
	if err := strictjson.Decode(raw, &value); err != nil {
		return nil, err
	}
	return &value, value.Validate()
}

func (c CostContribution) Validate() error {
	if len(c.ReportID) > 256 || c.ReceivedAt.IsZero() || !metricIntegerInRange(c.ModelCalls) ||
		(c.ReportID == "" && (c.Complete || c.ModelCalls != 0 || c.Cost != nil)) {
		return contracts.Invalidf("cost contribution identity is invalid")
	}
	if c.Cost != nil {
		return c.Cost.Validate()
	}
	return nil
}

// CostContributionChanged uses the same participant identity as report delivery.
// A replay never charges again; a complete contribution wins over a placeholder.
func CostContributionChanged(old *CostContribution, next CostContribution) (bool, error) {
	if err := next.Validate(); err != nil {
		return false, err
	}
	if old == nil {
		return true, nil
	}
	if err := old.Validate(); err != nil {
		return false, err
	}
	// Registration replay never replaces an observation. Conversely, a report
	// always fills an unreceived slot, even if its supplied time predates start.
	if next.ReportID == "" {
		return false, nil
	}
	if old.ReportID == "" {
		return true, nil
	}
	if old.Complete && !next.Complete {
		return false, nil
	}
	same := old.Complete == next.Complete && old.ModelCalls == next.ModelCalls && reflect.DeepEqual(old.Cost, next.Cost)
	if old.ReportID == next.ReportID {
		if !same {
			return false, contracts.Invalidf("cost report replay differs")
		}
		return false, nil
	}
	if old.Complete {
		if next.Complete && !same {
			return false, contracts.Invalidf("complete cost contribution differs")
		}
		return false, nil
	}
	if !next.Complete && !next.ReceivedAt.After(old.ReceivedAt) {
		return false, nil
	}
	return true, nil
}

func costGroupKey(g CostTotalGroup) string {
	return g.Model + "\x00" + g.Role + "\x00" + g.Phase + "\x00" + g.Source
}
func intValue(s string) *big.Int { v, _ := new(big.Int).SetString(s, 10); return v }

func costUnits(s string) *big.Int {
	r, _ := new(big.Rat).SetString(s)
	return r.Mul(r, new(big.Rat).SetInt(usdScale)).Num()
}

func changeCostNumber(target *string, change *big.Int, money bool) error {
	value := intValue(*target)
	if money {
		value = costUnits(*target)
	}
	value.Add(value, change)
	if value.Sign() < 0 {
		return contracts.Invalidf("cost total would be negative")
	}
	if money {
		*target = formatUSD(value)
	} else {
		*target = value.String()
	}
	return nil
}

// ReplaceCostContribution updates only bounded in-row totals. The composition
// owns one transaction across participant and parent stores; no history scan or
// additional receipt table is needed. Empty model denotes a combined remainder.
func (t *LLMCostTotals) ReplaceCostContribution(old, next *CostContribution, role string, inCheck bool, now time.Time) error {
	if err := t.Validate(); err != nil {
		return err
	}
	if role != "worker" && role != "planner" {
		return contracts.Invalidf("cost role is invalid")
	}
	for _, part := range []struct {
		value *CostContribution
		sign  int64
	}{{old, -1}, {next, 1}} {
		if part.value == nil {
			continue
		}
		c := part.value
		if err := c.Validate(); err != nil {
			return err
		}
		sign := part.sign
		if err := changeCostNumber(&t.Reports, big.NewInt(sign), false); err != nil {
			return err
		}
		if !c.Complete || c.Cost == nil {
			if err := changeCostNumber(&t.UnknownReports, big.NewInt(sign), false); err != nil {
				return err
			}
		}
		calls := c.ModelCalls
		if c.Cost != nil {
			calls = c.Cost.Calls
		}
		if err := changeCostNumber(&t.Calls, big.NewInt(sign*calls), false); err != nil {
			return err
		}
		if c.Cost == nil {
			continue
		}
		for _, g := range c.Cost.Groups {
			n := big.NewInt(sign * g.Calls)
			amount := costUnits(g.Amount)
			if sign < 0 {
				amount.Neg(amount)
			}
			for _, target := range []*string{&t.PricedCalls} {
				if err := changeCostNumber(target, n, false); err != nil {
					return err
				}
			}
			source := &t.ProviderCalls
			if g.Source == "gateway" {
				source = &t.GatewayCalls
			}
			if err := changeCostNumber(source, n, false); err != nil {
				return err
			}
			portion := &t.StandaloneAmount
			if inCheck {
				portion = &t.ChecksAmount
			}
			for _, target := range []*string{&t.Amount, portion} {
				if err := changeCostNumber(target, amount, true); err != nil {
					return err
				}
			}
			index := -1
			explicit := 0
			for i, prior := range t.Groups {
				if prior.Model != "" {
					explicit++
				}
				if prior.Model == g.Model && prior.Role == role && prior.Phase == g.Phase && prior.Source == g.Source {
					index = i
				}
			}
			if index < 0 {
				model := g.Model
				if explicit >= 48 {
					model = ""
				}
				for i, prior := range t.Groups {
					if prior.Model == model && prior.Role == role && prior.Phase == g.Phase && prior.Source == g.Source {
						index = i
					}
				}
				if index < 0 {
					t.Groups = append(t.Groups, CostTotalGroup{Model: model, Role: role, Phase: g.Phase, Source: g.Source, Calls: "0", Amount: "0"})
					index = len(t.Groups) - 1
				}
			}
			if err := changeCostNumber(&t.Groups[index].Calls, n, false); err != nil {
				return err
			}
			if err := changeCostNumber(&t.Groups[index].Amount, amount, true); err != nil {
				return err
			}
		}
	}
	// Expecting a report is not an execution observation. Keep the UI's last
	// observation absent until a real report arrives, and monotone thereafter.
	observed := (next != nil && next.ReportID != "") || (old != nil && old.ReportID != "")
	if observed && (t.UpdatedAt == nil || now.After(*t.UpdatedAt)) {
		t.UpdatedAt = &now
	}
	sort.Slice(t.Groups, func(i, j int) bool { return costGroupKey(t.Groups[i]) < costGroupKey(t.Groups[j]) })
	return t.Validate()
}

func (t LLMCostTotals) Encode() ([]byte, error) {
	if err := t.Validate(); err != nil {
		return nil, err
	}
	return json.Marshal(t)
}

func (t LLMCostTotals) KnownAmount() *string {
	if t.PricedCalls == "0" && (t.Reports == "0" || t.Calls != "0" || t.UnknownReports != "0" || t.HistoricalGap) {
		return nil
	}
	amount := t.Amount
	return &amount
}
