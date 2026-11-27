package controlplane

import (
	"fmt"
	"google.golang.org/grpc/codes"
)

type RuntimeAPIError struct {
	RPCStatus codes.Code
	Code      string
	Retryable bool
}

func (e *RuntimeAPIError) Error() string {
	return fmt.Sprintf("Runtime Agent returned gRPC %s (%s)", e.RPCStatus, e.Code)
}
