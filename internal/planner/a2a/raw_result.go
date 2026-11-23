package a2a

// The A2A SDK hands a DataPart to its caller as `any`, after encoding/json has
// turned every number into a float64 and kept only the last of duplicate
// keys. The Worker's WorkerCompletion is therefore decoded from the bytes of
// the JSON-RPC response that carried it: the bounded transport records the
// body of the latest response read under the invocation's context.

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"math"
	"strconv"
	"strings"
	"sync"
)

type responseCaptureKey struct{}

// responseCapture holds the body of the latest A2A response read under one
// context.
type responseCapture struct {
	mu   sync.Mutex
	body []byte
}

// withResponseCapture returns a context whose A2A calls record the body of
// their latest response in the returned capture.
func withResponseCapture(ctx context.Context) (context.Context, *responseCapture) {
	capture := &responseCapture{}
	return context.WithValue(ctx, responseCaptureKey{}, capture), capture
}

func responseCaptureFrom(ctx context.Context) *responseCapture {
	capture, _ := ctx.Value(responseCaptureKey{}).(*responseCapture)
	return capture
}

// record restarts the capture and returns body, which also records what its
// reader returns.
func (c *responseCapture) record(body io.ReadCloser) io.ReadCloser {
	c.mu.Lock()
	c.body = c.body[:0]
	c.mu.Unlock()
	return struct {
		io.Reader
		io.Closer
	}{Reader: io.TeeReader(body, c), Closer: body}
}

func (c *responseCapture) Write(data []byte) (int, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.body = append(c.body, data...)
	return len(data), nil
}

func (c *responseCapture) last() []byte {
	c.mu.Lock()
	defer c.mu.Unlock()
	return bytes.Clone(c.body)
}

// resultLocation is where the selected Worker message sits in a JSON-RPC
// result: message/send wraps a Message or a Task, tasks/get returns the Task,
// and a Task carries the message in its status or at a history index.
type resultLocation struct {
	wrapped bool
	task    bool
	history int // -1 selects Task.status.message
}

var errNoResultData = errors.New("A2A response has no Worker DataPart at the selected message")

// resultData returns the raw data member of the selected message's only part
// in a JSON-RPC response body. It reads the body as encoding/json does in the
// SDK, so both select the same message.
func resultData(body []byte, location resultLocation) ([]byte, error) {
	var response struct {
		Result json.RawMessage `json:"result"`
	}
	if err := json.NewDecoder(bytes.NewReader(body)).Decode(&response); err != nil {
		return nil, errNoResultData
	}
	value := response.Result
	if location.wrapped {
		var stream struct {
			Message json.RawMessage `json:"message"`
			Task    json.RawMessage `json:"task"`
		}
		if err := json.Unmarshal(value, &stream); err != nil {
			return nil, errNoResultData
		}
		value = stream.Message
		if location.task {
			value = stream.Task
		}
	}
	if location.task {
		var task struct {
			Status struct {
				Message json.RawMessage `json:"message"`
			} `json:"status"`
			History []json.RawMessage `json:"history"`
		}
		if err := json.Unmarshal(value, &task); err != nil {
			return nil, errNoResultData
		}
		switch {
		case location.history < 0:
			value = task.Status.Message
		case location.history < len(task.History):
			value = task.History[location.history]
		default:
			return nil, errNoResultData
		}
	}
	var message struct {
		Parts []struct {
			Data json.RawMessage `json:"data"`
		} `json:"parts"`
	}
	if err := json.Unmarshal(value, &message); err != nil || len(message.Parts) != 1 ||
		len(message.Parts[0].Data) == 0 {
		return nil, errNoResultData
	}
	return message.Parts[0].Data, nil
}

// integralNumbers rewrites every number in data that is written as a
// fraction or with an exponent but denotes an integral double, such as 7.0
// or 1e3, as the double's exact integer digits. An A2A DataPart is a
// google.protobuf.Value, whose numbers are doubles, and the Python Runtime
// renders each integer that way; an integer literal keeps its exact digits.
// A value outside the uint64 range is left as written and fails to decode.
func integralNumbers(data []byte) []byte {
	var result []byte
	copied := 0
	inString := false
	for index := 0; index < len(data); index++ {
		character := data[index]
		switch {
		case inString:
			if character == '\\' {
				index++
			} else if character == '"' {
				inString = false
			}
		case character == '"':
			inString = true
		case character == '-' || character >= '0' && character <= '9':
			end := index + 1
			for end < len(data) && strings.IndexByte("0123456789+-.eE", data[end]) >= 0 {
				end++
			}
			if digits, ok := integralDouble(data[index:end]); ok {
				result = append(result, data[copied:index]...)
				result = append(result, digits...)
				copied = end
			}
			index = end - 1
		}
	}
	if result == nil {
		return data
	}
	return append(result, data[copied:]...)
}

func integralDouble(token []byte) ([]byte, bool) {
	if bytes.IndexAny(token, ".eE") < 0 {
		return nil, false
	}
	value, err := strconv.ParseFloat(string(token), 64)
	if err != nil || value != math.Trunc(value) || math.Abs(value) >= 1<<64 {
		return nil, false
	}
	return strconv.AppendFloat(nil, value, 'f', 0, 64), true
}
