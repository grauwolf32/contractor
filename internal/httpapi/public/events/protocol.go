package events

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"regexp"
	"strconv"
	"time"

	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/runstore"
)

const (
	ProtocolVersion  = "contractor.events.v1"
	StreamRun        = "run"
	StreamOperations = "operations"

	maximumClientFrameBytes = 16 * 1024
	maximumServerFrameBytes = 64 * 1024
	maximumSubscriptions    = 32
)

var safeIdentifierPattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$`)
var subscriptionIDPattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$`)

type Stream struct {
	Kind string `json:"kind"`
	ID   string `json:"id,omitempty"`
}

type Cursor struct {
	Generation string `json:"generation"`
	Sequence   string `json:"sequence"`
}

type subscribeFrame struct {
	Version        string          `json:"version"`
	Type           string          `json:"type"`
	SubscriptionID string          `json:"subscriptionId"`
	Stream         json.RawMessage `json:"stream"`
	After          json.RawMessage `json:"after,omitempty"`
}

type unsubscribeFrame struct {
	Version        string `json:"version"`
	Type           string `json:"type"`
	SubscriptionID string `json:"subscriptionId"`
}

type clientFrame struct {
	typ            string
	subscriptionID string
	stream         Stream
	after          *Cursor
}

type subscribedFrame struct {
	Version        string `json:"version"`
	Type           string `json:"type"`
	SubscriptionID string `json:"subscriptionId"`
	Stream         Stream `json:"stream"`
	Cursor         Cursor `json:"cursor"`
}

type unsubscribedFrame struct {
	Version        string `json:"version"`
	Type           string `json:"type"`
	SubscriptionID string `json:"subscriptionId"`
}

type eventFrame struct {
	Version        string          `json:"version"`
	Type           string          `json:"type"`
	SubscriptionID string          `json:"subscriptionId"`
	Stream         Stream          `json:"stream"`
	Cursor         Cursor          `json:"cursor"`
	Kind           string          `json:"kind"`
	OccurredAt     time.Time       `json:"occurredAt"`
	Data           json.RawMessage `json:"data"`
}

type resyncRequiredFrame struct {
	Version        string `json:"version"`
	Type           string `json:"type"`
	SubscriptionID string `json:"subscriptionId"`
	Stream         Stream `json:"stream"`
	Reason         string `json:"reason"`
}

type errorFrame struct {
	Version        string `json:"version"`
	Type           string `json:"type"`
	SubscriptionID string `json:"subscriptionId,omitempty"`
	Code           string `json:"code"`
	Message        string `json:"message"`
	Retryable      bool   `json:"retryable"`
}

type operationsEventData struct {
	Resource   controlplane.OperationsResource `json:"resource"`
	ResourceID string                          `json:"resourceId,omitempty"`
	Revision   string                          `json:"revision"`
}

func decodeClientFrame(data []byte) (clientFrame, error) {
	if len(data) == 0 || len(data) > maximumClientFrameBytes {
		return clientFrame{}, errors.New("client frame exceeds its bounded contract")
	}
	var discriminator struct {
		Version string `json:"version"`
		Type    string `json:"type"`
	}
	if err := decodeClosedJSON(data, &discriminator, false); err != nil || discriminator.Version != ProtocolVersion {
		return clientFrame{}, errors.New("invalid protocol version or frame")
	}
	switch discriminator.Type {
	case "subscribe":
		var source subscribeFrame
		if err := decodeClosedJSON(data, &source, true); err != nil ||
			!subscriptionIDPattern.MatchString(source.SubscriptionID) {
			return clientFrame{}, errors.New("invalid subscribe frame")
		}
		stream, err := decodeStream(source.Stream)
		if err != nil {
			return clientFrame{}, err
		}
		var after *Cursor
		if len(source.After) != 0 {
			var cursor Cursor
			if err := decodeClosedJSON(source.After, &cursor, true); err != nil {
				return clientFrame{}, errors.New("invalid subscription cursor")
			}
			if _, err := parseCursor(cursor); err != nil {
				return clientFrame{}, errors.New("invalid subscription cursor")
			}
			after = &cursor
		}
		return clientFrame{
			typ: "subscribe", subscriptionID: source.SubscriptionID,
			stream: stream, after: after,
		}, nil
	case "unsubscribe":
		var source unsubscribeFrame
		if err := decodeClosedJSON(data, &source, true); err != nil ||
			!subscriptionIDPattern.MatchString(source.SubscriptionID) {
			return clientFrame{}, errors.New("invalid unsubscribe frame")
		}
		return clientFrame{typ: "unsubscribe", subscriptionID: source.SubscriptionID}, nil
	default:
		return clientFrame{}, errors.New("unknown client frame type")
	}
}

func decodeClosedJSON(data []byte, target any, disallowUnknown bool) error {
	decoder := json.NewDecoder(bytes.NewReader(data))
	if disallowUnknown {
		decoder.DisallowUnknownFields()
	}
	if err := decoder.Decode(target); err != nil {
		return err
	}
	var trailing any
	if err := decoder.Decode(&trailing); !errors.Is(err, io.EOF) {
		return errors.New("JSON frame contains trailing data")
	}
	return nil
}

func decodeStream(data json.RawMessage) (Stream, error) {
	var source struct {
		Kind string          `json:"kind"`
		ID   json.RawMessage `json:"id,omitempty"`
	}
	if err := decodeClosedJSON(data, &source, true); err != nil {
		return Stream{}, errors.New("invalid stream")
	}
	switch source.Kind {
	case StreamRun:
		var id string
		if len(source.ID) == 0 || decodeClosedJSON(source.ID, &id, true) != nil ||
			!safeIdentifierPattern.MatchString(id) {
			return Stream{}, errors.New("invalid Run stream")
		}
		return Stream{Kind: StreamRun, ID: id}, nil
	case StreamOperations:
		if len(source.ID) != 0 {
			return Stream{}, errors.New("Operations stream cannot contain an ID")
		}
		return Stream{Kind: StreamOperations}, nil
	default:
		return Stream{}, errors.New("unknown stream kind")
	}
}

func parseCursor(cursor Cursor) (uint64, error) {
	if !safeIdentifierPattern.MatchString(cursor.Generation) || len(cursor.Sequence) == 0 ||
		len(cursor.Sequence) > 20 || len(cursor.Sequence) > 1 && cursor.Sequence[0] == '0' {
		return 0, errors.New("invalid cursor")
	}
	sequence, err := strconv.ParseUint(cursor.Sequence, 10, 64)
	if err != nil {
		return 0, errors.New("invalid cursor sequence")
	}
	return sequence, nil
}

func runCursor(source runstore.WorkflowRunEventCursor) Cursor {
	return Cursor{Generation: source.Generation, Sequence: strconv.FormatInt(source.Sequence, 10)}
}

func operationsCursor(source controlplane.OperationsCursor) Cursor {
	return Cursor{Generation: source.Generation, Sequence: strconv.FormatUint(source.Revision, 10)}
}

func marshalServerFrame(frame any) ([]byte, error) {
	encoded, err := json.Marshal(frame)
	if err != nil {
		return nil, fmt.Errorf("encode Server event frame: %w", err)
	}
	if len(encoded) == 0 || len(encoded) > maximumServerFrameBytes {
		return nil, errors.New("Server event frame exceeds its bounded contract")
	}
	return encoded, nil
}

func runEventServerFrame(
	subscriptionID string,
	stream Stream,
	generation string,
	event runstore.WorkflowRunEvent,
) ([]byte, error) {
	kind := ""
	data, err := runstore.EncodePublicRunEventData(event)
	if err != nil {
		return nil, err
	}
	switch event.Kind {
	case runstore.RunEventLifecycleChanged:
		kind = "lifecycle.changed"
	default:
		kind = "planner.event"
	}
	return marshalServerFrame(eventFrame{
		Version: ProtocolVersion, Type: "event", SubscriptionID: subscriptionID,
		Stream: stream,
		Cursor: Cursor{Generation: generation, Sequence: strconv.FormatInt(event.SequenceNumber, 10)},
		Kind:   kind, OccurredAt: event.OccurredAt.UTC().Round(0), Data: data,
	})
}

func operationsEventServerFrame(
	subscriptionID string,
	stream Stream,
	change controlplane.OperationsChange,
) ([]byte, error) {
	data, err := json.Marshal(operationsEventData{
		Resource: change.Resource, ResourceID: change.ResourceID,
		Revision: strconv.FormatUint(change.Cursor.Revision, 10),
	})
	if err != nil {
		return nil, errors.New("encode Operations invalidation")
	}
	return marshalServerFrame(eventFrame{
		Version: ProtocolVersion, Type: "event", SubscriptionID: subscriptionID,
		Stream: stream, Cursor: operationsCursor(change.Cursor), Kind: "operations.changed",
		OccurredAt: change.OccurredAt.UTC().Round(0), Data: data,
	})
}
