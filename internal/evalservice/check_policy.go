package evalservice

// Human review is an explicit authenticated decision over an exact result and
// pinned rubric. Preparation may pin the protocol but never manufacture a
// verdict. Deterministic check evaluators are registered with V38-006.
func HumanPolicySHA256() string { return humanPolicyV1SHA256 }
