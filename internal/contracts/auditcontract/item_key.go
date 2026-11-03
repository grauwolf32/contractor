package auditcontract

import (
	"fmt"
	"strconv"
	"strings"
)

const (
	MaxExecutionMembers  = 64
	MaxFindingReferences = 128
)

// ExecutionItemKey is the stable key of a selected member in an ordinary Run.
// It is derived only from durable membership order, never from model output.
func ExecutionItemKey(ordinal int) string { return fmt.Sprintf("item-%02d", ordinal) }

func ExecutionItemOrdinal(key string) (int, bool) {
	value, err := strconv.Atoi(strings.TrimPrefix(key, "item-"))
	return value, err == nil && value >= 0 && value < MaxExecutionMembers && ExecutionItemKey(value) == key
}
