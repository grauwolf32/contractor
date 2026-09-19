package public

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/url"
	"strconv"

	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/grauwolf32/contractor/internal/evalstore"
	"github.com/grauwolf32/contractor/internal/requestid"
)

func (h *handler) evalError(w http.ResponseWriter, err error) {
	var d *evaldomain.Error
	if errors.As(err, &d) {
		writeJSON(w, d.Status, errorResponse{Code: d.Code, Message: d.Message, Retryable: d.Recovery == "retry_same_request", RequestID: requestid.FromResponse(w), Details: map[string]string{"kind": "eval", "recovery": d.Recovery}})
		return
	}
	h.handleError(w, err)
}
func (h *handler) evalJSON(w http.ResponseWriter, status int, kind string, v any) {
	raw, err := json.Marshal(v)
	if err == nil {
		err = evaldomain.Validate(kind, raw)
	}
	if err != nil {
		h.writeError(w, 500, "internal_error", "Evaluation response could not be rendered", false)
		return
	}
	writeJSON(w, status, json.RawMessage(raw))
}
func (h *handler) evalBody(w http.ResponseWriter, r *http.Request, kind string, cas bool) (evaldomain.Frozen, evaldomain.MutationIdentity, bool) {
	fail := func() (evaldomain.Frozen, evaldomain.MutationIdentity, bool) {
		h.evalError(w, evaldomain.Failure("eval_invalid"))
		return evaldomain.Frozen{}, evaldomain.MutationIdentity{}, false
	}
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		return fail()
	}
	media, err := requestMediaType(r)
	if err != nil || media != "application/json" {
		return fail()
	}
	if len(r.Header.Values("Idempotency-Key")) != 1 || len(r.Header.Values("If-Match")) > 1 || !cas && r.Header.Get("If-Match") != "" {
		return fail()
	}
	body, err := io.ReadAll(io.LimitReader(r.Body, evaldomain.MaxDocumentBytes+1))
	if err != nil {
		return fail()
	}
	doc, err := evaldomain.Freeze(kind, body)
	if err != nil {
		h.evalError(w, err)
		return doc, evaldomain.MutationIdentity{}, false
	}
	mutation, err := evaldomain.IdentifyMutation(r.Header.Get("Idempotency-Key"), r.Header.Get("If-Match"), cas, kind, body)
	if err != nil {
		h.evalError(w, err)
		return doc, mutation, false
	}
	return doc, mutation, true
}
func (h *handler) evalReceipt(w http.ResponseWriter, status int, r evalstore.Receipt) {
	ref, err := r.Experiment()
	if err != nil {
		h.handleError(w, err)
		return
	}
	if r.Replayed {
		w.Header().Set("Idempotency-Replayed", "true")
	}
	w.Header().Set("ETag", strconv.Quote(strconv.FormatInt(ref.Revision, 10)))
	h.evalJSON(w, status, "ExperimentReceipt", map[string]any{"experimentId": ref.ExperimentID, "revision": ref.Revision, "state": ref.State})
	if h.dependencies.EvalNotifier != nil {
		h.dependencies.EvalNotifier.Wake()
	}
}

func evalQuery(r *http.Request, extra ...string) (url.Values, int, error) {
	values, err := exactQuery(r.URL.RawQuery, append([]string{"limit", "cursor"}, extra...)...)
	if err != nil {
		return nil, 0, evaldomain.Failure("eval_invalid")
	}
	limit := 25
	if raw, ok := values["limit"]; ok {
		limit, err = strconv.Atoi(raw[0])
		if err != nil || limit < 1 || limit > 100 {
			return nil, 0, evaldomain.Failure("eval_invalid")
		}
	}
	if raw, ok := values["cursor"]; ok && (raw[0] == "" || len(raw[0]) > 8192) {
		return nil, 0, evaldomain.Failure("eval_invalid")
	}
	return values, limit, nil
}

type evalCursor struct {
	Context  string   `json:"context"`
	Position []string `json:"position"`
	Snapshot string   `json:"snapshot"`
}

func evalCursorContext(r *http.Request, q url.Values) string {
	copy := url.Values{}
	for k, v := range q {
		if k != "cursor" && k != "viewSnapshot" {
			copy[k] = v
		}
	}
	return evaldomain.Digest([]byte(principalUserID(r.Context()) + "\n" + r.URL.Path + "\n" + copy.Encode()))
}
func (h *handler) readEvalCursor(r *http.Request, q url.Values, count int) (evalCursor, error) {
	var out evalCursor
	if q.Get("cursor") == "" {
		return out, nil
	}
	b, err := base64.RawURLEncoding.DecodeString(q.Get("cursor"))
	if err != nil || len(b) <= sha256.Size {
		return out, evaldomain.Failure("eval_invalid")
	}
	raw, sig := b[:len(b)-sha256.Size], b[len(b)-sha256.Size:]
	mac := hmac.New(sha256.New, h.tokenDigest[:])
	mac.Write(raw)
	if !hmac.Equal(sig, mac.Sum(nil)) || json.Unmarshal(raw, &out) != nil || out.Context != evalCursorContext(r, q) || len(out.Position) != count || out.Snapshot == "" {
		return out, evaldomain.Failure("eval_invalid")
	}
	return out, nil
}
func (h *handler) evalPage(r *http.Request, q url.Values, more bool, snapshot string, positions ...string) (evalPageInfo, error) {
	out := evalPageInfo{HasMore: more}
	if !more {
		return out, nil
	}
	raw, err := json.Marshal(evalCursor{evalCursorContext(r, q), positions, snapshot})
	if err != nil {
		return out, err
	}
	mac := hmac.New(sha256.New, h.tokenDigest[:])
	mac.Write(raw)
	cursor := base64.RawURLEncoding.EncodeToString(append(raw, mac.Sum(nil)...))
	if len(cursor) > 8192 {
		return out, evaldomain.Failure("eval_limit_exceeded")
	}
	out.NextCursor = &cursor
	return out, nil
}
func cursorRevision(c evalCursor) (*int64, error) {
	if c.Snapshot == "" {
		return nil, nil
	}
	revision, err := strconv.ParseInt(c.Snapshot, 10, 64)
	if err != nil || revision < 0 {
		return nil, evaldomain.Failure("eval_invalid")
	}
	return &revision, nil
}
