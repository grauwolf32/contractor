package runtimewire

import (
	"encoding/json"
	"github.com/grauwolf32/contractor/internal/contracts"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protoreflect"
)

// Encode projects a validated domain request into its named protobuf message.
// Canonical documents remain JSON; RPC identity and operation fields are typed.
func Encode(value contracts.Validatable, target proto.Message) error {
	if err := value.Validate(); err != nil {
		return ErrInvalid
	}
	var raw []byte
	var err error
	if private, ok := value.(interface{ RevealJSON() ([]byte, error) }); ok {
		raw, err = private.RevealJSON()
	} else {
		raw, err = json.Marshal(value)
	}
	defer clear(raw)
	if err != nil || len(raw) > MaxControlBytes {
		return ErrInvalid
	}
	if err := encodeMessage(raw, target.ProtoReflect()); err != nil {
		return ErrInvalid
	}
	if proto.Size(target) > MaxControlBytes {
		return ErrInvalid
	}
	return nil
}

// Decode applies the existing strict domain parser. The caller validates after
// its policy-specific telemetry salvage, just as at the persisted report boundary.
func Decode(message proto.Message, target contracts.Validatable) error {
	if proto.Size(message) > MaxControlBytes {
		return ErrInvalid
	}
	raw, err := decodeMessage(message.ProtoReflect())
	if err != nil || len(raw) > MaxControlBytes {
		return ErrInvalid
	}
	if err := contracts.DecodePrivate(raw, target); err != nil {
		return ErrInvalid
	}
	return nil
}

// Clear releases an allocation request's private document bytes after its call.
func Clear(message proto.Message) {
	var wipe func(protoreflect.Message)
	wipe = func(m protoreflect.Message) {
		m.Range(func(f protoreflect.FieldDescriptor, v protoreflect.Value) bool {
			if f.Kind() == protoreflect.BytesKind {
				clear(v.Bytes())
			} else if f.Kind() == protoreflect.MessageKind {
				if f.IsMap() {
					return true
				}
				if f.IsList() {
					list := v.List()
					for i := 0; i < list.Len(); i++ {
						wipe(list.Get(i).Message())
					}
				} else {
					wipe(v.Message())
				}
			}
			return true
		})
	}
	wipe(message.ProtoReflect())
	proto.Reset(message)
}
