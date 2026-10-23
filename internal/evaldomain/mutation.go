package evaldomain

import (
	"encoding/json"
	"strconv"

	"github.com/grauwolf32/contractor/internal/contracts"
)

// MutationIdentity is persisted with a command. It is intentionally separate
// from portable document hashes, which always hash exact retained bytes.
type MutationIdentity struct {
	Key              string  `json:"key"`
	RequestSHA256    string  `json:"requestSha256"`
	ExpectedRevision *uint64 `json:"expectedRevision"`
}

// MutationPrecondition checks the headers of a Managed Eval mutation: its
// key, which must be in the public Idempotency-Key grammar, and its If-Match,
// which a state transition requires. It returns the expected revision, if
// any. It needs no body, so the API checks it before reading one.
func MutationPrecondition(key, ifMatch string, requireRevision bool) (*uint64, error) {
	if !contracts.ValidIdempotencyKey(key) {
		return nil, Failure("eval_invalid")
	}
	if ifMatch == "" {
		if requireRevision {
			return nil, Failure("eval_precondition_required")
		}
		return nil, nil
	}
	if len(ifMatch) < 3 || ifMatch[0] != '"' || ifMatch[len(ifMatch)-1] != '"' {
		return nil, Failure("eval_invalid")
	}
	n, err := strconv.ParseUint(ifMatch[1:len(ifMatch)-1], 10, 64)
	if err != nil || n == 0 {
		return nil, Failure("eval_invalid")
	}
	return &n, nil
}

// IdentifyMutation binds a Managed Eval mutation to its key and precondition
// (MutationPrecondition) and its exact body.
func IdentifyMutation(key, ifMatch string, requireRevision bool, kind string, body []byte) (MutationIdentity, error) {
	revision, err := MutationPrecondition(key, ifMatch, requireRevision)
	if err != nil {
		return MutationIdentity{}, err
	}
	if err := Validate(kind, body); err != nil {
		return MutationIdentity{}, err
	}
	value, err := StrictJSON(body)
	if err != nil {
		return MutationIdentity{}, err
	}
	canonical, err := json.Marshal(struct {
		Kind     string
		Body     any
		Revision *uint64
	}{kind, value, revision})
	if err != nil {
		return MutationIdentity{}, Failure("eval_invalid")
	}
	return MutationIdentity{Key: key, RequestSHA256: Digest(canonical), ExpectedRevision: revision}, nil
}

// CheckMutation accepts an exact replay before checking a now-stale revision.
// The caller first scopes stored receipts by authenticated owner and resource.
func CheckMutation(incoming MutationIdentity, stored *MutationIdentity, currentRevision uint64) (bool, error) {
	if stored != nil && incoming.Key == stored.Key {
		if incoming.RequestSHA256 != stored.RequestSHA256 {
			return false, Failure("eval_idempotency_conflict")
		}
		return true, nil
	}
	if incoming.ExpectedRevision != nil && *incoming.ExpectedRevision != currentRevision {
		return false, Failure("eval_revision_mismatch")
	}
	return false, nil
}
