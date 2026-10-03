package evalstore

type ListParams struct {
	OwnerID, ProjectID, State, DatasetID, ControlMode, AfterID string
	Limit                                                      int
	Revision                                                   *int64
}

type ExperimentSummary struct {
	ID          string `json:"experimentId"`
	ProjectID   string `json:"projectId"`
	Name        string `json:"name"`
	ControlMode string `json:"controlMode"`
	State       string `json:"state"`
	Revision    int64  `json:"revision"`
	Expected    int    `json:"expectedMembers"`
}
