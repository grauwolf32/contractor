//go:build e2e

package e2e

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"net/http"

	"github.com/grauwolf32/contractor/api/v1alpha1/runtimepb"
	"github.com/grauwolf32/contractor/internal/runtimewire"
	"google.golang.org/protobuf/proto"
)

// Inspect the native Artifact metadata at the existing response-proxy boundary.
// Reads replay only their first bounded frame, leaving chunks streamed; writes
// join the complete unary receipt and its successful trailer before faulting it.
func privateArtifactRPCMetadata(response *http.Response) (*runtimepb.ArtifactPayloadMetadata, error) {
	if response.StatusCode != http.StatusOK {
		return nil, nil
	}
	path := response.Request.URL.Path
	if path != runtimepb.Artifact_Read_FullMethodName && path != runtimepb.Artifact_Write_FullMethodName {
		return nil, nil
	}
	var raw []byte
	if path == runtimepb.Artifact_Write_FullMethodName {
		var err error
		raw, err = io.ReadAll(io.LimitReader(response.Body, runtimewire.BulkFrameBytes+6))
		_ = response.Body.Close()
		response.Body = io.NopCloser(bytes.NewReader(raw))
		if err != nil {
			return nil, err
		}
		if response.Trailer.Get("Grpc-Status") != "0" {
			return nil, nil
		}
	} else {
		var header [5]byte
		if _, err := io.ReadFull(response.Body, header[:]); err != nil {
			if err == io.EOF {
				return nil, nil
			}
			return nil, err
		}
		size := binary.BigEndian.Uint32(header[1:])
		if header[0] != 0 || size > runtimewire.BulkFrameBytes {
			return nil, fmt.Errorf("invalid Artifact proxy frame header")
		}
		raw = make([]byte, 5+int(size))
		copy(raw, header[:])
		if _, err := io.ReadFull(response.Body, raw[5:]); err != nil {
			return nil, err
		}
		response.Body = struct {
			io.Reader
			io.Closer
		}{io.MultiReader(bytes.NewReader(raw), response.Body), response.Body}
	}
	if len(raw) < 5 || len(raw) > runtimewire.BulkFrameBytes+5 || raw[0] != 0 || int(binary.BigEndian.Uint32(raw[1:5])) != len(raw)-5 {
		return nil, fmt.Errorf("invalid Artifact proxy metadata frame")
	}
	if path == runtimepb.Artifact_Write_FullMethodName {
		var metadata runtimepb.ArtifactPayloadMetadata
		if err := proto.Unmarshal(raw[5:], &metadata); err != nil {
			return nil, err
		}
		return &metadata, nil
	}
	var frame runtimepb.ArtifactReadFrame
	if err := proto.Unmarshal(raw[5:], &frame); err != nil {
		return nil, err
	}
	if frame.GetHeader() == nil {
		return nil, fmt.Errorf("Artifact proxy read started without metadata")
	}
	return frame.GetHeader(), nil
}
