package reporting

import (
	"math/big"
	"regexp"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/grauwolf32/contractor/internal/contracts"
)

const MaxLLMCostGroups = 64

var usdCanonical = regexp.MustCompile(`^(0|[1-9][0-9]{0,17})(\.[0-9]{0,17}[1-9])?$`)
var usdScale = new(big.Int).Exp(big.NewInt(10), big.NewInt(18), nil)

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

func formatUSD(units *big.Int) string {
	whole, fraction := new(big.Int), new(big.Int)
	whole.QuoRem(units, usdScale, fraction)
	if fraction.Sign() == 0 {
		return whole.String()
	}
	tail := fraction.String()
	return whole.String() + "." + strings.TrimRight(strings.Repeat("0", 18-len(tail))+tail, "0")
}
