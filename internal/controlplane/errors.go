package controlplane

import "errors"

var (
	ErrInvalidRequest       = errors.New("invalid Control Plane request")
	ErrRegistrationConflict = errors.New("Runtime Agent registration conflicts with an existing process identity")
	ErrHeartbeatOutOfOrder  = errors.New("Runtime Agent heartbeat is out of order")
	ErrInsufficientCapacity = errors.New("insufficient compatible Runtime Agent capacity")
	ErrReservationConflict  = errors.New("Stage reservation conflicts with its original request")
	ErrReservationReleased  = errors.New("Stage reservation has already been released")
	ErrAllocationNotFound   = errors.New("active allocation not found")
	ErrAgentNotFound        = errors.New("Runtime Agent not found")
	ErrOperationsGeneration = errors.New("Operations cursor generation changed")
	ErrOperationsCursor     = errors.New("Operations cursor is unavailable")
	ErrOperationsGap        = errors.New("Operations cursor has a sequence gap")
)
