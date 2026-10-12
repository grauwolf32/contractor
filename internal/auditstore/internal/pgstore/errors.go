package pgstore

import (
	"fmt"

	"github.com/grauwolf32/contractor/internal/auditstore"
)

func invalidf(format string, arguments ...any) error {
	return fmt.Errorf("%w: %s", auditstore.ErrInvalid, fmt.Sprintf(format, arguments...))
}
