package public

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/url"
	"sort"

	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/grauwolf32/contractor/internal/evalservice"
	"github.com/grauwolf32/contractor/internal/evalstore"
)

type evalBin struct {
	Context  string              `json:"context"`
	Snapshot string              `json:"snapshot"`
	Suite    string              `json:"suite"`
	Bin      evalstore.BinFilter `json:"bin"`
}

func evalBinContext(r *http.Request) string {
	return evaldomain.Digest([]byte(principalUserID(r.Context()) + "\n" + r.PathValue("id")))
}

func (h *handler) evalBinMAC(raw []byte) []byte {
	mac := hmac.New(sha256.New, h.tokenDigest[:])
	mac.Write([]byte("eval-bin/v1\n"))
	mac.Write(raw)
	return mac.Sum(nil)
}

func (h *handler) evalBinToken(r *http.Request, chart evalservice.ChartView, bin evaldomain.Bin) (string, error) {
	suite := ""
	if chart.SuiteID != nil {
		suite = *chart.SuiteID
	}
	raw, err := json.Marshal(evalBin{Context: evalBinContext(r), Snapshot: chart.Snapshot, Suite: suite, Bin: evalstore.BinFilter{Metric: chart.Chart, Scope: *chart.MeasurementScope, Lower: bin.Lower, Upper: bin.Upper, UpperInclusive: bin.UpperInclusive}})
	if err != nil {
		return "", err
	}
	return base64.RawURLEncoding.EncodeToString(append(raw, h.evalBinMAC(raw)...)), nil
}

func (h *handler) readEvalBin(r *http.Request, q url.Values, p *evalservice.MemberPageParams) error {
	if len(q.Get("binFilter")) > maxEvalCursorBytes {
		return evaldomain.Failure("eval_invalid")
	}
	encoded, err := base64.RawURLEncoding.DecodeString(q.Get("binFilter"))
	if err != nil || len(encoded) <= sha256.Size {
		return evaldomain.Failure("eval_invalid")
	}
	raw, signature := encoded[:len(encoded)-sha256.Size], encoded[len(encoded)-sha256.Size:]
	var token evalBin
	if !hmac.Equal(signature, h.evalBinMAC(raw)) || json.Unmarshal(raw, &token) != nil || token.Context != evalBinContext(r) || token.Suite != p.SuiteID || p.Filter != "" || p.VariantID != "" || p.MeasurementScope != "" && p.MeasurementScope != token.Bin.Scope {
		return evaldomain.Failure("eval_invalid")
	}
	if p.Snapshot != "" && p.Snapshot != token.Snapshot {
		return evaldomain.Failure("eval_view_changed")
	}
	p.Snapshot, p.MeasurementScope, p.Bin = token.Snapshot, token.Bin.Scope, &token.Bin
	return nil
}

func sortedEvalKeys[T any](values map[string]T) []string {
	keys := make([]string, 0, len(values))
	for key := range values {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}
