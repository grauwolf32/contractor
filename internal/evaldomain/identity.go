package evaldomain

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"regexp"
)

var idPattern = regexp.MustCompile(`^[a-z0-9][a-z0-9._-]{0,127}$`)
var memberPattern = regexp.MustCompile(`^[0-9a-f]{64}$`)
var digestPattern = regexp.MustCompile(`^sha256:[0-9a-f]{64}$`)

func Digest(data []byte) string {
	h := sha256.Sum256(data)
	return "sha256:" + hex.EncodeToString(h[:])
}

func MemberID(experiment, suite, caseID string, sample int, variant string) (string, error) {
	if !idPattern.MatchString(variant) {
		return "", Failure("eval_invalid")
	}
	if err := pairIdentityValid(experiment, suite, caseID, sample); err != nil {
		return "", err
	}
	return identityHash([]any{experiment, suite, caseID, sample, variant}), nil
}

func PairID(experiment, suite, caseID string, sample int) (string, error) {
	if err := pairIdentityValid(experiment, suite, caseID, sample); err != nil {
		return "", err
	}
	return identityHash([]any{experiment, suite, caseID, sample}), nil
}

func pairIdentityValid(experiment, suite, caseID string, sample int) error {
	if !idPattern.MatchString(experiment) || !idPattern.MatchString(suite) || !idPattern.MatchString(caseID) || sample < 1 || sample > MaxRepetitions {
		return Failure("eval_invalid")
	}
	return nil
}

func identityHash(parts []any) string {
	data, _ := json.Marshal(parts)
	return Digest(data)[len("sha256:"):]
}
