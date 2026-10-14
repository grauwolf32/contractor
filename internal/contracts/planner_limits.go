package contracts

import (
	"fmt"

	"github.com/grauwolf32/contractor/internal/strictjson"
)

const (
	DefaultPlannerSubtasks = 32
	MaxPlannerSubtasks     = 1000
)

// PlannerSubtaskLimit is an optional Workflow plan-size selection. Its zero
// value represents omission, whose contract-defined effective value is 32.
// Explicit JSON values must be positive; null is not an omission.
type PlannerSubtaskLimit int

func (limit PlannerSubtaskLimit) Effective() int {
	if limit == 0 {
		return DefaultPlannerSubtasks
	}
	return int(limit)
}

func (limit PlannerSubtaskLimit) Validate() error {
	if limit < 0 || limit > MaxPlannerSubtasks {
		return fmt.Errorf("maxSubtasks must be an integer from 1 to %d when present", MaxPlannerSubtasks)
	}
	return nil
}

func (limit *PlannerSubtaskLimit) UnmarshalJSON(data []byte) error {
	var value *int
	if err := strictjson.Decode(data, &value); err != nil {
		return fmt.Errorf("maxSubtasks: %w", err)
	}
	if value == nil || *value <= 0 || *value > MaxPlannerSubtasks {
		return fmt.Errorf("maxSubtasks must be an integer from 1 to %d", MaxPlannerSubtasks)
	}
	*limit = PlannerSubtaskLimit(*value)
	return nil
}
