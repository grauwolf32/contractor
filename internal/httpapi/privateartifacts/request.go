package privateartifacts

import (
	"errors"
	"fmt"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
)

func decodeRequestJSON(data []byte, target any) error {
	if strictjson.HasNULEscape(data) {
		return fmt.Errorf("%w: JSON body contains a NUL escape", errInvalidRequest)
	}
	if err := strictjson.Decode(data, target); errors.Is(err, strictjson.ErrTrailingData) {
		return fmt.Errorf("%w: JSON body has trailing data", errInvalidRequest)
	} else if err != nil {
		return fmt.Errorf("%w: decode JSON body", errInvalidRequest)
	}
	return nil
}
