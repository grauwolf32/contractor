package controlplane

import (
	"encoding/json"

	"github.com/grauwolf32/contractor/internal/contracts/control"
)

// encodeLifecycleRequest encodes one Runtime Agent lifecycle request body. The
// prepare request is the one private wire message that carries RuntimeSettings
// secrets, which default JSON encoding redacts (SEC-R88), so it uses their
// explicit allocation-wire encoder; postJSON clears the body after sending.
func encodeLifecycleRequest(request any) ([]byte, error) {
	if prepare, ok := request.(control.PrepareAllocationRequest); ok {
		return prepare.RevealJSON()
	}
	return json.Marshal(request)
}
