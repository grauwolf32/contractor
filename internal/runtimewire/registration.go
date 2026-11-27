// Package runtimewire adapts generated private protobuf messages to the existing
// domain contracts without changing their presence, grammar or numeric rules.
package runtimewire

import (
	"encoding/json"
	"errors"

	pb "github.com/grauwolf32/contractor/api/v1alpha1/runtimepb"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/control"
	"google.golang.org/protobuf/proto"
)

const (
	MaxControlBytes   = 1 << 20
	MaxA2ABytes       = 4 << 20
	MaxMetadataBytes  = 16 << 10
	MaxConcurrentRPCs = 32
)

var ErrInvalid = errors.New("private protobuf value violates its contract")

func Registration(request *pb.AgentRegistration) (control.AgentRegistration, error) {
	if request == nil || proto.Size(request) > MaxControlBytes {
		return control.AgentRegistration{}, ErrInvalid
	}
	data, err := decodeMessage(request.ProtoReflect())
	if err != nil || len(data) > MaxControlBytes {
		return control.AgentRegistration{}, ErrInvalid
	}
	value, err := contracts.DecodePrivateStrict[control.AgentRegistration](data)
	if err != nil {
		return control.AgentRegistration{}, ErrInvalid
	}
	return value, nil
}

func RegistrationResponse(value control.AgentRegistrationResponse) (*pb.AgentRegistrationResponse, error) {
	if err := value.Validate(); err != nil {
		return nil, ErrInvalid
	}
	data, err := json.Marshal(value)
	if err != nil {
		return nil, ErrInvalid
	}
	response := new(pb.AgentRegistrationResponse)
	if err := encodeMessage(data, response.ProtoReflect()); err != nil {
		return nil, ErrInvalid
	}
	return response, nil
}
