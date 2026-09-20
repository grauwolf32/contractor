package auditpriority

import (
	"crypto/sha256"
	"encoding/hex"
	"regexp"
	"sort"
	"strings"
	"unicode/utf8"

	"github.com/grauwolf32/contractor/internal/contracts"
)

var identifierPattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._:-]*$`)
var digestPattern = regexp.MustCompile(`^sha256:[0-9a-f]{64}$`)

func validIdentifier(value string) bool {
	return len(value) <= MaxIdentifierBytes && identifierPattern.MatchString(value)
}

func validText(value string, maximum int) bool {
	return len(value) <= maximum && utf8.ValidString(value) && !strings.ContainsRune(value, 0) && strings.TrimSpace(value) != ""
}

func digestBytes(data []byte) string {
	sum := sha256.Sum256(data)
	return "sha256:" + hex.EncodeToString(sum[:])
}

// CandidateID depends on the exact inventory and original checklist identity,
// never on input position, model score, Round, Run or descriptive title.
func CandidateID(inventoryDigest, itemKey, itemVersion string) (string, error) {
	if !digestPattern.MatchString(inventoryDigest) || !validIdentifier(itemKey) || !validText(itemVersion, MaxItemVersionBytes) {
		return "", invalid(CodeInvalidPool)
	}
	data, err := contracts.MarshalPrivateCanonical(struct {
		Schema          string `json:"schema"`
		InventoryDigest string `json:"inventory_digest"`
		ItemKey         string `json:"item_key"`
		ItemVersion     string `json:"item_version"`
	}{"contractor.audit.priority-candidate-id.v1", inventoryDigest, itemKey, itemVersion})
	if err != nil {
		return "", invalid(CodeInvalidPool)
	}
	return "priority-candidate-" + strings.TrimPrefix(digestBytes(data), "sha256:"), nil
}

// NewPool normalizes candidate order without changing identity or deduplicating
// distinct items. Duplicate checklist keys, including differing versions, fail.
// Empty remaining pools are allowed; initial empty-inventory policy is external.
func NewPool(inventoryDigest string, items []ItemIdentity) (Pool, error) {
	if !digestPattern.MatchString(inventoryDigest) || len(items) > MaxCandidates {
		return Pool{}, invalid(CodeInvalidPool)
	}
	pool := Pool{Schema: PoolSchema, InventoryDigest: inventoryDigest, Candidates: make([]Candidate, 0, len(items))}
	seen := make(map[string]bool, len(items))
	for _, item := range items {
		id, err := CandidateID(inventoryDigest, item.Key, item.Version)
		if err != nil || seen[item.Key] {
			return Pool{}, invalid(CodeInvalidPool)
		}
		seen[item.Key] = true
		pool.Candidates = append(pool.Candidates, Candidate{ID: id, ItemKey: item.Key, ItemVersion: item.Version})
	}
	sort.Slice(pool.Candidates, func(i, j int) bool { return pool.Candidates[i].ID < pool.Candidates[j].ID })
	return pool, nil
}

func (p Pool) Validate() error {
	if p.Schema != PoolSchema || !digestPattern.MatchString(p.InventoryDigest) || p.Candidates == nil || len(p.Candidates) > MaxCandidates {
		return invalid(CodeInvalidPool)
	}
	previous := ""
	seen := make(map[string]bool, len(p.Candidates))
	for _, candidate := range p.Candidates {
		id, err := CandidateID(p.InventoryDigest, candidate.ItemKey, candidate.ItemVersion)
		if err != nil || id != candidate.ID || id <= previous || seen[candidate.ItemKey] {
			return invalid(CodeInvalidPool)
		}
		previous, seen[candidate.ItemKey] = id, true
	}
	return nil
}

func MarshalPool(pool Pool) ([]byte, error) {
	if err := pool.Validate(); err != nil {
		return nil, err
	}
	data, err := contracts.MarshalPrivateCanonical(pool)
	if err != nil || len(data) > MaxSelectionBytes {
		return nil, invalid(CodeInvalidPool)
	}
	return data, nil
}

func PoolDigest(pool Pool) (string, error) {
	data, err := MarshalPool(pool)
	if err != nil {
		return "", err
	}
	return digestBytes(data), nil
}
