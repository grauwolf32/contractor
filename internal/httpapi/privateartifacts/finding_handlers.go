package privateartifacts

import "github.com/grauwolf32/contractor/internal/findingintake"

func findingSubmissionResponse(
	receipt findingintake.Receipt,
	replayed bool,
) findingintake.SubmissionResponse {
	evidence := receipt.Evidence
	if evidence == nil {
		evidence = []findingintake.ExactArtifact{}
	}
	return findingintake.SubmissionResponse{
		FindingID:  receipt.FindingID,
		APIVersion: findingintake.APIVersion,
		ProposalID: receipt.ProposalID,
		ReceiptID:  receipt.ReceiptID,
		Proposal:   receipt.Proposal,
		Evidence:   evidence,
		Replayed:   replayed,
	}
}
