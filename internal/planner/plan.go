package planner

import (
	"fmt"
	"reflect"
	"strconv"
	"strings"
	"sync"

	"github.com/grauwolf32/contractor/internal/contracts"
)

const (
	MaxPlannerSubtasks                = 32
	MaxPlannerGlobalObjectiveBytes    = 64 * 1024
	MaxPlannerSubtaskObjectiveBytes   = 8 * 1024
	MaxPlannerSubtaskInstructionBytes = 32 * 1024
	maxPlannerLogicalWorkerNameBytes  = 128
)

type PlannerSubtaskStatus string

const (
	PlannerSubtaskPending   PlannerSubtaskStatus = "pending"
	PlannerSubtaskRunning   PlannerSubtaskStatus = "running"
	PlannerSubtaskSucceeded PlannerSubtaskStatus = "succeeded"
	PlannerSubtaskFailed    PlannerSubtaskStatus = "failed"
)

type PlannerSubtask struct {
	ID           string               `json:"id"`
	Objective    string               `json:"objective"`
	Instructions string               `json:"instructions"`
	Status       PlannerSubtaskStatus `json:"status"`
}

type PlannerActiveDispatch struct {
	CallID     string `json:"callId"`
	SubtaskID  string `json:"subtaskId"`
	WorkerName string `json:"workerName"`
}

// PlannerPlan is the bounded typed projection of one model-backed Planner
// invocation. Objective is copied from the immutable Stage. Subtask text is
// accepted only through the explicit bounded fields; identity and every status
// transition are controlled by PlannerPlanController.
type PlannerPlan struct {
	Revision         uint64                 `json:"revision"`
	Objective        string                 `json:"objective"`
	Subtasks         []PlannerSubtask       `json:"subtasks"`
	CurrentSubtaskID string                 `json:"currentSubtaskId,omitempty"`
	ActiveDispatch   *PlannerActiveDispatch `json:"activeDispatch,omitempty"`
}

// PlannerPlanProjection is the durable/UI-safe modeled-Planner state. The
// immutable Stage objective is intentionally absent and must be joined by the
// caller from the StageExecution snapshot.
type PlannerPlanProjection struct {
	Revision         uint64                 `json:"revision"`
	Subtasks         []PlannerSubtask       `json:"subtasks"`
	CurrentSubtaskID string                 `json:"currentSubtaskId,omitempty"`
	ActiveDispatch   *PlannerActiveDispatch `json:"activeDispatch,omitempty"`
}

type PlanError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

func (e *PlanError) Error() string { return e.Code + ": " + e.Message }

type PlannerDispatchClaim struct {
	CallID  string
	Subtask PlannerSubtask
}

// PlannerPlanController owns the live first-slice plan state. Its adapter
// persists validated projections through PlanSessionService; callers can only
// receive deep snapshots and cannot mutate the authoritative plan directly.
type PlannerPlanController struct {
	mu sync.Mutex

	plan             PlannerPlan
	nextSubtaskID    uint64
	nextDispatchCall uint64
}

func NewPlannerPlanController(objective string) (*PlannerPlanController, error) {
	if err := validatePlanText(
		"global objective", objective, MaxPlannerGlobalObjectiveBytes,
	); err != nil {
		return nil, err
	}
	return &PlannerPlanController{plan: PlannerPlan{
		Objective: objective,
		Subtasks:  []PlannerSubtask{},
	}}, nil
}

func (c *PlannerPlanController) Snapshot() PlannerPlan {
	c.mu.Lock()
	defer c.mu.Unlock()
	return clonePlannerPlan(c.plan)
}

func (c *PlannerPlanController) AddSubtask(
	objective string, instructions string,
) (PlannerPlan, *PlanError) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.plan.ActiveDispatch != nil {
		return clonePlannerPlan(c.plan), planError(
			"planner_plan_dispatch_active", "Cannot change the plan while a Worker dispatch is active",
		)
	}
	if len(c.plan.Subtasks) >= MaxPlannerSubtasks {
		return clonePlannerPlan(c.plan), planError(
			"planner_subtask_limit", "Planner reached its subtask limit",
		)
	}
	if err := validatePlanText(
		"subtask objective", objective, MaxPlannerSubtaskObjectiveBytes,
	); err != nil {
		return clonePlannerPlan(c.plan), planError("planner_subtask_invalid", err.Error())
	}
	if err := validatePlanText(
		"subtask instructions", instructions, MaxPlannerSubtaskInstructionBytes,
	); err != nil {
		return clonePlannerPlan(c.plan), planError("planner_subtask_invalid", err.Error())
	}
	id := strconv.FormatUint(c.nextSubtaskID, 10)
	c.nextSubtaskID++
	c.plan.Subtasks = append(c.plan.Subtasks, PlannerSubtask{
		ID: id, Objective: objective, Instructions: instructions, Status: PlannerSubtaskPending,
	})
	if c.plan.CurrentSubtaskID == "" {
		c.plan.CurrentSubtaskID = id
	}
	c.plan.Revision++
	return clonePlannerPlan(c.plan), nil
}

// CurrentSubtask validates the exact model-supplied ID without reserving a
// dispatch. It lets adapters validate the remaining request before the atomic
// claim, while ClaimCurrentSubtask still fences a concurrent winner.
func (c *PlannerPlanController) CurrentSubtask(subtaskID string) (PlannerSubtask, *PlanError) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.plan.ActiveDispatch != nil {
		return PlannerSubtask{}, planError(
			"planner_dispatch_active", "Another Worker dispatch is already active",
		)
	}
	if c.plan.CurrentSubtaskID == "" || subtaskID != c.plan.CurrentSubtaskID {
		return PlannerSubtask{}, planError(
			"planner_subtask_stale", "subtask_id does not identify the current pending subtask",
		)
	}
	index := c.subtaskIndexLocked(subtaskID)
	if index < 0 || c.plan.Subtasks[index].Status != PlannerSubtaskPending {
		return PlannerSubtask{}, planError(
			"planner_subtask_stale", "subtask_id does not identify the current pending subtask",
		)
	}
	return c.plan.Subtasks[index], nil
}

// ClaimCurrentSubtask atomically validates and marks the exact current pending
// subtask. The generated call ID fences late results from an older dispatch.
func (c *PlannerPlanController) ClaimCurrentSubtask(
	subtaskID string, workerName string,
) (PlannerDispatchClaim, *PlanError) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.plan.ActiveDispatch != nil {
		return PlannerDispatchClaim{}, planError(
			"planner_dispatch_active", "Another Worker dispatch is already active",
		)
	}
	if c.plan.CurrentSubtaskID == "" {
		return PlannerDispatchClaim{}, planError(
			"planner_subtask_unavailable", "The plan has no current pending subtask",
		)
	}
	if subtaskID != c.plan.CurrentSubtaskID {
		return PlannerDispatchClaim{}, planError(
			"planner_subtask_stale", "subtask_id does not identify the current pending subtask",
		)
	}
	index := c.subtaskIndexLocked(subtaskID)
	if index < 0 || c.plan.Subtasks[index].Status != PlannerSubtaskPending {
		return PlannerDispatchClaim{}, planError(
			"planner_subtask_stale", "subtask_id does not identify the current pending subtask",
		)
	}
	if strings.TrimSpace(workerName) == "" || len(workerName) > maxPlannerLogicalWorkerNameBytes {
		return PlannerDispatchClaim{}, planError(
			"planner_worker_invalid", "Logical Worker name is invalid",
		)
	}
	c.nextDispatchCall++
	callID := fmt.Sprintf("dispatch-%04d", c.nextDispatchCall)
	c.plan.Subtasks[index].Status = PlannerSubtaskRunning
	c.plan.ActiveDispatch = &PlannerActiveDispatch{
		CallID: callID, SubtaskID: subtaskID, WorkerName: workerName,
	}
	c.plan.Revision++
	return PlannerDispatchClaim{
		CallID: callID, Subtask: c.plan.Subtasks[index],
	}, nil
}

func (c *PlannerPlanController) CompleteDispatch(
	callID string, outcome contracts.StageOutcome,
) (PlannerPlan, *PlanError) {
	switch outcome {
	case contracts.StageSucceeded:
		return c.resolveDispatch(callID, PlannerSubtaskSucceeded)
	case contracts.StageFailed:
		return c.resolveDispatch(callID, PlannerSubtaskFailed)
	default:
		return c.Snapshot(), planError(
			"planner_dispatch_result_invalid", "Worker result has an unknown outcome",
		)
	}
}

// FailDispatch records a bounded dispatch failure when no valid Worker
// WorkerResult exists. It clears the active claim and advances the plan.
func (c *PlannerPlanController) FailDispatch(callID string) (PlannerPlan, *PlanError) {
	return c.resolveDispatch(callID, PlannerSubtaskFailed)
}

func (c *PlannerPlanController) CanFinish(outcome contracts.StageOutcome) *PlanError {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.plan.ActiveDispatch != nil {
		return planError(
			"planner_finish_dispatch_active", "Cannot finish while a Worker dispatch is active",
		)
	}
	switch outcome {
	case contracts.StageFailed:
		return nil
	case contracts.StageSucceeded:
		hasSucceeded := false
		for _, subtask := range c.plan.Subtasks {
			switch subtask.Status {
			case PlannerSubtaskSucceeded:
				hasSucceeded = true
			case PlannerSubtaskPending, PlannerSubtaskRunning:
				return planError(
					"planner_finish_incomplete", "Cannot finish successfully while planned work remains",
				)
			}
		}
		if !hasSucceeded {
			return planError(
				"planner_finish_incomplete", "Successful finish requires at least one succeeded subtask",
			)
		}
		return nil
	default:
		return planError("planner_finish_invalid", "Finish outcome is invalid")
	}
}

func (c *PlannerPlanController) resolveDispatch(
	callID string, status PlannerSubtaskStatus,
) (PlannerPlan, *PlanError) {
	c.mu.Lock()
	defer c.mu.Unlock()
	active := c.plan.ActiveDispatch
	if active == nil || active.CallID != callID {
		return clonePlannerPlan(c.plan), planError(
			"planner_dispatch_stale", "Worker result does not match the active dispatch",
		)
	}
	index := c.subtaskIndexLocked(active.SubtaskID)
	if index < 0 || c.plan.CurrentSubtaskID != active.SubtaskID ||
		c.plan.Subtasks[index].Status != PlannerSubtaskRunning {
		return clonePlannerPlan(c.plan), planError(
			"planner_dispatch_stale", "Worker result does not match the current running subtask",
		)
	}
	c.plan.Subtasks[index].Status = status
	c.plan.ActiveDispatch = nil
	c.plan.CurrentSubtaskID = ""
	for next := index + 1; next < len(c.plan.Subtasks); next++ {
		if c.plan.Subtasks[next].Status == PlannerSubtaskPending {
			c.plan.CurrentSubtaskID = c.plan.Subtasks[next].ID
			break
		}
	}
	c.plan.Revision++
	return clonePlannerPlan(c.plan), nil
}

func (c *PlannerPlanController) subtaskIndexLocked(id string) int {
	for index := range c.plan.Subtasks {
		if c.plan.Subtasks[index].ID == id {
			return index
		}
	}
	return -1
}

func (p PlannerPlan) Validate() error {
	if err := validatePlanText(
		"global objective", p.Objective, MaxPlannerGlobalObjectiveBytes,
	); err != nil {
		return err
	}
	return p.Projection().Validate()
}

func (p PlannerPlan) Projection() PlannerPlanProjection {
	return clonePlannerPlanProjection(PlannerPlanProjection{
		Revision: p.Revision, Subtasks: p.Subtasks,
		CurrentSubtaskID: p.CurrentSubtaskID, ActiveDispatch: p.ActiveDispatch,
	})
}

func (p PlannerPlanProjection) Validate() error {
	if len(p.Subtasks) > MaxPlannerSubtasks {
		return fmt.Errorf("Planner plan exceeds its subtask limit")
	}
	currentIndex := -1
	firstUnresolved := -1
	running := 0
	for index, subtask := range p.Subtasks {
		if subtask.ID != strconv.Itoa(index) {
			return fmt.Errorf("Planner subtask ID/order is invalid")
		}
		if err := validatePlanText(
			"subtask objective", subtask.Objective, MaxPlannerSubtaskObjectiveBytes,
		); err != nil {
			return err
		}
		if err := validatePlanText(
			"subtask instructions", subtask.Instructions, MaxPlannerSubtaskInstructionBytes,
		); err != nil {
			return err
		}
		switch subtask.Status {
		case PlannerSubtaskPending, PlannerSubtaskSucceeded, PlannerSubtaskFailed:
		case PlannerSubtaskRunning:
			running++
		default:
			return fmt.Errorf("Planner subtask status is invalid")
		}
		if subtask.ID == p.CurrentSubtaskID {
			currentIndex = index
		}
		if firstUnresolved < 0 &&
			(subtask.Status == PlannerSubtaskPending || subtask.Status == PlannerSubtaskRunning) {
			firstUnresolved = index
		}
	}
	if p.CurrentSubtaskID == "" {
		if firstUnresolved >= 0 {
			return fmt.Errorf("Planner plan omits a current unresolved subtask")
		}
	} else if currentIndex < 0 ||
		currentIndex != firstUnresolved ||
		(p.Subtasks[currentIndex].Status != PlannerSubtaskPending &&
			p.Subtasks[currentIndex].Status != PlannerSubtaskRunning) {
		return fmt.Errorf("Planner current subtask is invalid")
	}
	if p.ActiveDispatch == nil {
		if running != 0 {
			return fmt.Errorf("Planner running subtask has no active dispatch")
		}
		return nil
	}
	if running != 1 || currentIndex < 0 ||
		p.Subtasks[currentIndex].Status != PlannerSubtaskRunning ||
		p.ActiveDispatch.SubtaskID != p.CurrentSubtaskID ||
		strings.TrimSpace(p.ActiveDispatch.CallID) == "" ||
		strings.TrimSpace(p.ActiveDispatch.WorkerName) == "" ||
		len(p.ActiveDispatch.WorkerName) > maxPlannerLogicalWorkerNameBytes {
		return fmt.Errorf("Planner active dispatch is invalid")
	}
	return nil
}

// ValidatePlannerPlanTransition rejects a stored projection history that
// could not have been produced by PlannerPlanController. Text and IDs are
// immutable after append; only adapter-owned status/current/dispatch fields
// may change through their matching event kind.
func ValidatePlannerPlanTransition(
	previous *PlannerPlanProjection,
	next PlannerPlanProjection,
	kind PlannerEventKind,
) error {
	if err := next.Validate(); err != nil {
		return err
	}
	baseline := PlannerPlanProjection{Subtasks: []PlannerSubtask{}}
	if previous != nil {
		baseline = clonePlannerPlanProjection(*previous)
		if err := baseline.Validate(); err != nil {
			return fmt.Errorf("previous Planner plan projection is invalid: %w", err)
		}
	}
	if next.Revision != baseline.Revision+1 {
		return fmt.Errorf("Planner plan revision must advance by exactly one")
	}
	switch kind {
	case PlannerEventPlanChanged:
		return validatePlanAppendTransition(baseline, next)
	case PlannerEventDispatchStarted:
		return validateDispatchStartTransition(baseline, next)
	case PlannerEventDispatchCompleted:
		return validateDispatchCompletionTransition(baseline, next)
	default:
		return fmt.Errorf("event kind %q cannot change a Planner plan", kind)
	}
}

func validatePlanAppendTransition(previous, next PlannerPlanProjection) error {
	if previous.ActiveDispatch != nil || next.ActiveDispatch != nil ||
		len(next.Subtasks) != len(previous.Subtasks)+1 {
		return fmt.Errorf("plan.changed must append exactly one idle subtask")
	}
	for index := range previous.Subtasks {
		if !reflect.DeepEqual(previous.Subtasks[index], next.Subtasks[index]) {
			return fmt.Errorf("plan.changed modified an existing subtask")
		}
	}
	appended := next.Subtasks[len(next.Subtasks)-1]
	if appended.Status != PlannerSubtaskPending || appended.ID != strconv.Itoa(len(previous.Subtasks)) {
		return fmt.Errorf("plan.changed appended subtask identity or status is invalid")
	}
	wantCurrent := previous.CurrentSubtaskID
	if wantCurrent == "" {
		wantCurrent = appended.ID
	}
	if next.CurrentSubtaskID != wantCurrent {
		return fmt.Errorf("plan.changed selected an invalid current subtask")
	}
	return nil
}

func validateDispatchStartTransition(previous, next PlannerPlanProjection) error {
	if previous.ActiveDispatch != nil || next.ActiveDispatch == nil ||
		previous.CurrentSubtaskID == "" || next.CurrentSubtaskID != previous.CurrentSubtaskID ||
		len(previous.Subtasks) != len(next.Subtasks) {
		return fmt.Errorf("dispatch.started has an invalid plan shape")
	}
	changed := 0
	for index := range previous.Subtasks {
		before, after := previous.Subtasks[index], next.Subtasks[index]
		if before.ID != after.ID || before.Objective != after.Objective ||
			before.Instructions != after.Instructions {
			return fmt.Errorf("dispatch.started modified immutable subtask data")
		}
		if before.Status != after.Status {
			if before.ID != previous.CurrentSubtaskID || before.Status != PlannerSubtaskPending ||
				after.Status != PlannerSubtaskRunning {
				return fmt.Errorf("dispatch.started has an invalid status transition")
			}
			changed++
		}
	}
	if changed != 1 || next.ActiveDispatch.SubtaskID != next.CurrentSubtaskID {
		return fmt.Errorf("dispatch.started did not claim exactly the current subtask")
	}
	return nil
}

func validateDispatchCompletionTransition(previous, next PlannerPlanProjection) error {
	if previous.ActiveDispatch == nil || next.ActiveDispatch != nil ||
		len(previous.Subtasks) != len(next.Subtasks) {
		return fmt.Errorf("dispatch.completed has an invalid plan shape")
	}
	completedIndex := -1
	for index := range previous.Subtasks {
		before, after := previous.Subtasks[index], next.Subtasks[index]
		if before.ID != after.ID || before.Objective != after.Objective ||
			before.Instructions != after.Instructions {
			return fmt.Errorf("dispatch.completed modified immutable subtask data")
		}
		if before.Status == after.Status {
			continue
		}
		if before.ID != previous.ActiveDispatch.SubtaskID || before.Status != PlannerSubtaskRunning ||
			(after.Status != PlannerSubtaskSucceeded && after.Status != PlannerSubtaskFailed) {
			return fmt.Errorf("dispatch.completed has an invalid status transition")
		}
		if completedIndex >= 0 {
			return fmt.Errorf("dispatch.completed changed more than one subtask")
		}
		completedIndex = index
	}
	if completedIndex < 0 || previous.CurrentSubtaskID != previous.ActiveDispatch.SubtaskID {
		return fmt.Errorf("dispatch.completed did not resolve its active subtask")
	}
	wantCurrent := ""
	for index := completedIndex + 1; index < len(next.Subtasks); index++ {
		if next.Subtasks[index].Status == PlannerSubtaskPending {
			wantCurrent = next.Subtasks[index].ID
			break
		}
	}
	if next.CurrentSubtaskID != wantCurrent {
		return fmt.Errorf("dispatch.completed selected an invalid next current subtask")
	}
	return nil
}

func validatePlanText(name string, value string, limit int) error {
	if strings.TrimSpace(value) == "" {
		return fmt.Errorf("%s must not be empty", name)
	}
	if len(value) > limit {
		return fmt.Errorf("%s exceeds %d bytes", name, limit)
	}
	return nil
}

func clonePlannerPlan(input PlannerPlan) PlannerPlan {
	result := input
	result.Subtasks = append([]PlannerSubtask(nil), input.Subtasks...)
	if input.ActiveDispatch != nil {
		active := *input.ActiveDispatch
		result.ActiveDispatch = &active
	}
	return result
}

func clonePlannerPlanProjection(input PlannerPlanProjection) PlannerPlanProjection {
	result := input
	result.Subtasks = append([]PlannerSubtask(nil), input.Subtasks...)
	if input.ActiveDispatch != nil {
		active := *input.ActiveDispatch
		result.ActiveDispatch = &active
	}
	return result
}

func planError(code string, message string) *PlanError {
	return &PlanError{Code: code, Message: message}
}
