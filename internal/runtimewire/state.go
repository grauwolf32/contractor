package runtimewire

import (
	"encoding/json"
	"fmt"

	pb "github.com/grauwolf32/contractor/api/v1alpha1/runtimepb"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/reporting"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	"google.golang.org/protobuf/proto"
)

// The domain snapshot retains its 4 MiB ceiling. The separate wire ceiling
// leaves room for its revision and protobuf envelope without shrinking it.
const MaxStateMessageBytes = reporting.MaxAgentStateSnapshotBytes + 1024

func WorkerStateResult(message *pb.StateReadResponse, conditional uint64) (reporting.WorkerStateReadResult, error) {
	invalid := reporting.WorkerStateReadResult{}
	if message == nil || proto.Size(message) > MaxStateMessageBytes {
		return invalid, ErrInvalid
	}
	raw, err := decodeMessage(message.ProtoReflect())
	if err != nil {
		return invalid, ErrInvalid
	}
	var value struct {
		Revision    uint64          `json:"revision" strictjson:"required"`
		Snapshot    json.RawMessage `json:"snapshot,omitempty"`
		NotModified *struct{}       `json:"notModified,omitempty"`
	}
	if strictjson.Decode(raw, &value) != nil || value.Revision == 0 {
		return invalid, ErrInvalid
	}
	etag := fmt.Sprintf("\"contractor-agent-state-v1-%d\"", value.Revision)
	if value.NotModified != nil {
		if conditional == 0 || value.Revision != conditional || value.Snapshot != nil {
			return invalid, ErrInvalid
		}
		return reporting.WorkerStateReadResult{ETag: etag, NotModified: true}, nil
	}
	if len(value.Snapshot) == 0 || len(value.Snapshot) > reporting.MaxAgentStateSnapshotBytes || value.Revision == conditional {
		return invalid, ErrInvalid
	}
	snapshot, err := contracts.DecodePrivateStrict[reporting.AgentStateSnapshot](value.Snapshot)
	if err != nil || snapshot.State.StateRevision != value.Revision {
		return invalid, ErrInvalid
	}
	return reporting.WorkerStateReadResult{ETag: etag, Snapshot: &snapshot}, nil
}
