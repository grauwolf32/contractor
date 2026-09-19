package auditstore

import (
	"encoding/json"
)

type executionMemberJSON struct {
	ExecutionItemID string          `json:"execution_item_id"`
	ItemID          string          `json:"item_id"`
	BatchOrdinal    int             `json:"batch_ordinal"`
	ItemAttempt     int             `json:"item_attempt"`
	TaskRef         json.RawMessage `json:"task_ref"`
	TaskDigest      string          `json:"task_digest"`
	Inputs          []ExactArtifact `json:"inputs"`
}

type executionIntentWrite struct {
	manifestRef, members json.RawMessage
	roundID              *string
	roleAttempt          *int
}

// prepareExecutionIntentWrite detaches optional values and serializes validated
// members without changing the order, exact artifact refs or empty-array shape.
func prepareExecutionIntentWrite(params CreateExecutionIntentParams) executionIntentWrite {
	manifestRef, _ := json.Marshal(params.Manifest.Ref)
	members := make([]executionMemberJSON, len(params.Members))
	for index, member := range params.Members {
		taskRef, _ := json.Marshal(member.Task.Ref)
		members[index] = executionMemberJSON{
			ExecutionItemID: member.ExecutionItemID, ItemID: member.ItemID,
			BatchOrdinal: member.BatchOrdinal, ItemAttempt: member.ItemAttempt,
			TaskRef: taskRef, TaskDigest: member.Task.Digest,
			Inputs: append([]ExactArtifact{}, member.Inputs...),
		}
	}
	encodedMembers, _ := json.Marshal(members)
	var roundID *string
	if params.RoundID != nil {
		value := *params.RoundID
		roundID = &value
	}
	var roleAttempt *int
	if params.RoleAttempt != nil {
		value := *params.RoleAttempt
		roleAttempt = &value
	}
	return executionIntentWrite{manifestRef: manifestRef, members: encodedMembers, roundID: roundID, roleAttempt: roleAttempt}
}
