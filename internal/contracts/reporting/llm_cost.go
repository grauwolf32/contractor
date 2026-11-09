package reporting

import (
	"math/big"
	"regexp"
	"sort"
	"strings"
	"sync"
	"unicode"
	"unicode/utf8"

	"github.com/grauwolf32/contractor/internal/contracts"
)

const MaxLLMCostGroups = 64

var usdCanonical = regexp.MustCompile(`^(0|[1-9][0-9]{0,17})(\.[0-9]{0,17}[1-9])?$`)
var usdNumber = regexp.MustCompile(`^(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]{1,2})?$`)
var usdScale = new(big.Int).Exp(big.NewInt(10), big.NewInt(18), nil)
var usdMaximum = new(big.Int).Exp(big.NewInt(10), big.NewInt(36), nil)

type LLMCostGroup struct {
	Model  string `json:"model" strictjson:"required"`
	Phase  string `json:"phase" strictjson:"required"`
	Source string `json:"source" strictjson:"required"`
	Calls  int64  `json:"calls" strictjson:"required"`
	Amount string `json:"amount" strictjson:"required"`
}

// LLMCost preserves exact USD observations; unpriced calls are calls minus
// the sum of group calls. It is independent of logical budget accounting.
type LLMCost struct {
	Currency string         `json:"currency" strictjson:"required"`
	Calls    int64          `json:"calls" strictjson:"required"`
	Groups   []LLMCostGroup `json:"groups" strictjson:"required"`
}

func (c LLMCost) Validate() error {
	if c.Currency != "USD" || !metricIntegerInRange(c.Calls) || c.Groups == nil || len(c.Groups) > MaxLLMCostGroups {
		return contracts.Invalidf("LLM cost shape is invalid")
	}
	seen := make(map[string]bool, len(c.Groups))
	var priced int64
	for _, g := range c.Groups {
		if !validCostGroup(g) || !usdCanonical.MatchString(g.Amount) || g.Calls <= 0 || g.Calls > c.Calls-priced {
			return contracts.Invalidf("LLM cost group is invalid")
		}
		key := g.Model + "\x00" + g.Phase + "\x00" + g.Source
		if seen[key] {
			return contracts.Invalidf("LLM cost group is duplicated")
		}
		seen[key] = true
		priced += g.Calls
	}
	return nil
}

func validCostGroup(g LLMCostGroup) bool {
	if g.Model == "" || strings.TrimSpace(g.Model) != g.Model || !utf8.ValidString(g.Model) || len(g.Model) > 256 {
		return false
	}
	for _, r := range g.Model {
		if unicode.IsControl(r) {
			return false
		}
	}
	if g.Source != "provider" && g.Source != "gateway" {
		return false
	}
	switch g.Phase {
	case "main", "compaction", "finalization", "summarization":
		return true
	}
	return false
}

// NormalizeUSD accepts a bounded provider JSON number or cost header and
// returns its exact canonical decimal. It never rounds an observation.
func NormalizeUSD(raw string) (string, bool) {
	if len(raw) > 80 || !usdNumber.MatchString(raw) {
		return "", false
	}
	r, ok := new(big.Rat).SetString(raw)
	if !ok {
		return "", false
	}
	r.Mul(r, new(big.Rat).SetInt(usdScale))
	if !r.IsInt() || r.Sign() < 0 || r.Num().Cmp(usdMaximum) >= 0 {
		return "", false
	}
	return formatUSD(r.Num()), true
}

func formatUSD(units *big.Int) string {
	whole, fraction := new(big.Int), new(big.Int)
	whole.QuoRem(units, usdScale, fraction)
	if fraction.Sign() == 0 {
		return whole.String()
	}
	tail := fraction.String()
	return whole.String() + "." + strings.TrimRight(strings.Repeat("0", 18-len(tail))+tail, "0")
}

// AddUSD sums two canonical bounded amounts without binary floating point.
func AddUSD(a, b string) (string, bool) {
	if !usdCanonical.MatchString(a) || !usdCanonical.MatchString(b) {
		return "", false
	}
	left, _ := new(big.Rat).SetString(a)
	right, _ := new(big.Rat).SetString(b)
	left.Add(left, right).Mul(left, new(big.Rat).SetInt(usdScale))
	if left.Num().Cmp(usdMaximum) >= 0 {
		return "", false
	}
	return formatUSD(left.Num()), true
}

// LLMCostCollector is one existing invocation/report owner's bounded reducer.
// It stores no transport objects or provider content.
type LLMCostCollector struct {
	mu     sync.Mutex
	calls  int64
	groups []LLMCostGroup
}

func (c *LLMCostCollector) Observe(model, phase, source, raw string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.calls == contracts.MaxExactInteger {
		return
	}
	c.calls++
	amount, ok := NormalizeUSD(raw)
	g := LLMCostGroup{Model: model, Phase: phase, Source: source, Calls: 1, Amount: amount}
	if !ok || !validCostGroup(g) {
		return
	}
	for i := range c.groups {
		prior := &c.groups[i]
		if prior.Model == model && prior.Phase == phase && prior.Source == source {
			if sum, valid := AddUSD(prior.Amount, amount); valid {
				prior.Amount = sum
				prior.Calls++
			}
			return
		}
	}
	if len(c.groups) < MaxLLMCostGroups {
		c.groups = append(c.groups, g)
	}
}

func (c *LLMCostCollector) Snapshot(minCalls int64) *LLMCost {
	c.mu.Lock()
	defer c.mu.Unlock()
	groups := append([]LLMCostGroup{}, c.groups...)
	sort.Slice(groups, func(i, j int) bool {
		a, b := groups[i], groups[j]
		if a.Model != b.Model {
			return a.Model < b.Model
		}
		if a.Phase != b.Phase {
			return a.Phase < b.Phase
		}
		return a.Source < b.Source
	})
	return &LLMCost{Currency: "USD", Calls: min(contracts.MaxExactInteger, max(c.calls, minCalls)), Groups: groups}
}
