package evalservice

import (
	_ "embed"
	"github.com/grauwolf32/contractor/internal/evaldomain"
)

// Human review is an explicit authenticated decision over an exact result and
// pinned rubric. Preparation may pin the protocol but never manufacture a
// verdict. Deterministic check evaluators are registered with V38-006.
//
//go:embed check_policy.go
var humanPolicySource []byte

func HumanPolicySHA256() string { return evaldomain.Digest(humanPolicySource) }
