package auditcontract

import "github.com/grauwolf32/contractor/internal/helpers/strictjson"

const (
	TaskSchema         = "contractor.audit.task.v1"
	MaxMutationRecords = 128
	MaxMutationBytes   = 1 << 20
)

// Task is accepted ordinary work. Its identities and exact input selections
// come from the Audit owner, rather than the producer of a work proposal.
type Task struct {
	Schema     string              `json:"schema"`
	AuditID    string              `json:"audit_id"`
	TaskID     string              `json:"task_id"`
	Workflow   string              `json:"workflow"`
	Inputs     map[string]Material `json:"inputs"`
	Parameters map[string]string   `json:"parameters"`
	Hypothesis *HypothesisRef      `json:"hypothesis,omitempty"`
	Objective  string              `json:"objective"`
}

type HypothesisRef struct {
	ID       string `json:"id"`
	Revision uint64 `json:"revision"`
}

func DecodeTask(data []byte) (Task, error) {
	canonical, err := Decode(data, nil)
	if err != nil {
		return Task{}, err
	}
	var task Task
	if strictjson.Decode(canonical, &task) != nil || task.Schema != TaskSchema {
		return Task{}, errInvalid
	}
	return task, nil
}
