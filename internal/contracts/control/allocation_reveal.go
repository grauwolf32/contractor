package control

import "encoding/json"

// RevealJSON encodes the prepare request, the one private wire message that
// carries RuntimeSettings secrets in plaintext (SEC-R29, SEC-R88). Default
// JSON encoding of the request redacts them.
func (r PrepareAllocationRequest) RevealJSON() ([]byte, error) {
	spec, err := r.Spec.RevealJSON()
	if err != nil {
		return nil, err
	}
	type plain PrepareAllocationRequest
	return json.Marshal(struct {
		plain
		Spec json.RawMessage `json:"spec"`
	}{plain(r), spec})
}

// RevealJSON encodes the spec with its RuntimeSettings secrets in plaintext,
// for PrepareAllocationRequest.RevealJSON.
func (s AllocationSpec) RevealJSON() ([]byte, error) {
	settings, err := s.RuntimeSettings.RevealJSON()
	if err != nil {
		return nil, err
	}
	type plain AllocationSpec
	return json.Marshal(struct {
		plain
		RuntimeSettings json.RawMessage `json:"runtimeSettings"`
	}{plain(s), settings})
}
