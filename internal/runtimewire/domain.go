package runtimewire

import (
	"encoding/json"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	"google.golang.org/protobuf/proto"
)

// EncodeDomain preserves the owning domain's compact-JSON byte limit while
// separately bounding its named protobuf envelope. Callers own semantic checks.
func EncodeDomain(value any, target proto.Message, domainLimit, wireLimit int) error {
	raw, err := contracts.EncodeCompactJSON(value)
	if err != nil || len(raw) > domainLimit {
		return ErrInvalid
	}
	if encodeMessage(raw, target.ProtoReflect()) != nil || proto.Size(target) > wireLimit {
		return ErrInvalid
	}
	return nil
}

// DecodeDomain rejects unknown fields recursively and preserves presence and
// integer digits. The existing domain owner still validates all invariants.
func DecodeDomain(message proto.Message, target any, domainLimit, wireLimit int) error {
	if message == nil || proto.Size(message) > wireLimit {
		return ErrInvalid
	}
	raw, err := decodeMessage(message.ProtoReflect())
	if err != nil || strictjson.HasNULEscape(raw) {
		return ErrInvalid
	}
	size, err := contracts.ResultJSONSize(json.RawMessage(raw))
	if err != nil || size > domainLimit || strictjson.DecodeNonNull(raw, target) != nil {
		return ErrInvalid
	}
	return nil
}
