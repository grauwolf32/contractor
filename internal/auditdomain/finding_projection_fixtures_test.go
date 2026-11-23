package auditdomain

func testProjectedFindingProposal() FindingProposal {
	return FindingProposal{Schema: FindingProposalSchema, ClientKey: "candidate-a", Title: "Authorization gap", Description: "Ownership validation may be missing.",
		Subject: &FindingSubject{Kind: "openapi-operation", Key: "get-widget"}, Hypothesis: "A caller may access another owner's widget.",
		Preconditions: []string{}, StandardRefs: []StandardReference{}, EvidenceIDs: []string{},
		ProposedChecks:     []ProposedCheck{{Objective: "Trace the ownership predicate.", Method: "static-trace"}, {Objective: "Confirm behavior against the authorized target.", Method: "http-probe"}},
		SeveritySuggestion: "medium", Limitations: []string{"partial-trace", "needs-live-confirmation"}}
}
