package privateartifacts

// This test-only recorder keeps the domain assertions of the Artifact tests.
// It translates their request vocabulary to actual protobuf frames and runs
// the production gRPC server; no private HTTP handler is retained or called.
import (
	"bytes"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	artifactpostgres "github.com/grauwolf32/contractor/internal/artifacts/postgres"
	"github.com/grauwolf32/contractor/internal/helpers/requestid"
	"github.com/grauwolf32/contractor/internal/httpapi/httpx"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"strings"
	"time"

	pb "github.com/grauwolf32/contractor/api/v1alpha1/runtimepb"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	"github.com/grauwolf32/contractor/internal/runtimewire"
	rpcstatus "google.golang.org/genproto/googleapis/rpc/status"
	"google.golang.org/grpc/codes"
	"google.golang.org/protobuf/proto"
)

const bindingCreatedAtHeader = "X-Contractor-Binding-Created-At"
const revisionCreatedAtHeader = "X-Contractor-Revision-Created-At"

func newPayloadNativeRecorder(dependencies Dependencies) (http.Handler, error) {
	native, err := NewGRPCHandler(dependencies)
	if err != nil {
		return nil, err
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		parts := strings.Split(strings.TrimPrefix(r.URL.Path, "/private/v1/allocations/"), "/")
		if len(parts) < 2 {
			http.NotFound(w, r)
			return
		}
		target := pb.Artifact_Read_FullMethodName
		if parts[1] == "finding-proposals" {
			target = pb.Artifact_SubmitFinding_FullMethodName
		} else if r.Method == http.MethodPut {
			target = pb.Artifact_Write_FullMethodName
		}
		original := r
		request := r.Clone(r.Context())
		request.URL.Path = target
		request.URL.RawPath = ""
		request.URL.RawQuery = ""
		request.Method = http.MethodPost
		request.Proto = "HTTP/2.0"
		request.ProtoMajor = 2
		request.ProtoMinor = 0
		request.Header.Set("Content-Type", "application/grpc")
		request.Header.Set("Te", "trailers")
		request.Header.Set(allocationIDHeader, parts[0])
		request.ContentLength = -1
		body := &lazyPayloadFrames{build: func() (io.Reader, error) { return payloadFixtureFrames(original, parts, target) }}
		request.Body = io.NopCloser(body)
		response := &nativePayloadRecorder{ResponseRecorder: httptest.NewRecorder(), outer: w}
		native.ServeHTTP(response, request)
		writePayloadFixtureResponse(w, response, original, target)
	}), nil
}

type nativePayloadRecorder struct {
	*httptest.ResponseRecorder
	outer http.ResponseWriter
}

func (r *nativePayloadRecorder) Unwrap() http.ResponseWriter { return r.outer }

type lazyPayloadFrames struct {
	build  func() (io.Reader, error)
	reader io.Reader
	err    error
}

func (r *lazyPayloadFrames) Read(p []byte) (int, error) {
	if r.build != nil {
		r.reader, r.err = r.build()
		r.build = nil
	}
	if r.err != nil {
		return 0, r.err
	}
	return r.reader.Read(p)
}
func fixtureFrame(value proto.Message) []byte {
	data, err := proto.Marshal(value)
	if err != nil {
		panic(err)
	}
	framed := make([]byte, len(data)+5)
	binary.BigEndian.PutUint32(framed[1:5], uint32(len(data)))
	copy(framed[5:], data)
	return framed
}
func fixtureUnknown(value proto.Message) { value.ProtoReflect().SetUnknown([]byte{0xf8, 0x07, 0x01}) }

func payloadFixtureFrames(r *http.Request, parts []string, target string) (io.Reader, error) {
	if target == pb.Artifact_SubmitFinding_FullMethodName {
		raw, err := io.ReadAll(io.LimitReader(r.Body, artifacts.MaxPayloadSize+1))
		if err != nil {
			return nil, err
		}
		var fields map[string]json.RawMessage
		if strictjson.Decode(raw, &fields) != nil {
			return bytes.NewReader(fixtureFrame(&pb.FindingSubmissionFrame{Frame: &pb.FindingSubmissionFrame_Header{Header: new(pb.FindingSubmissionHeader)}})), nil
		}
		var docs [][]byte
		for _, field := range []struct{ name, size string }{{"proposal", "proposalSize"}, {"derived_anchors", "derivedAnchorsSize"}, {"httpEvidence", "httpEvidenceSize"}} {
			data := fields[field.name]
			docs = append(docs, data)
			delete(fields, field.name)
			if data != nil {
				fields[field.size] = json.RawMessage(strconv.Itoa(len(data)))
			}
		}
		header := new(pb.FindingSubmissionHeader)
		if runtimewire.EncodeDomain(fields, header, runtimewire.BulkHeaderBytes, runtimewire.BulkFrameBytes) != nil {
			fixtureUnknown(header)
		}
		if r.URL.RawQuery != "" || r.Header.Get("Content-Type") != "application/json" {
			fixtureUnknown(header)
		}
		frames := []io.Reader{bytes.NewReader(fixtureFrame(&pb.FindingSubmissionFrame{Frame: &pb.FindingSubmissionFrame_Header{Header: header}}))}
		for i, doc := range docs {
			for offset := 0; offset < len(doc); {
				size := runtimewire.BulkChunkSize(len(doc) - offset)
				frames = append(frames, bytes.NewReader(fixtureFrame(&pb.FindingSubmissionFrame{Frame: &pb.FindingSubmissionFrame_Chunk{Chunk: &pb.FindingDocumentChunk{Document: pb.FindingDocumentKind(i + 1).Enum(), Data: doc[offset : offset+size]}}})))
				offset += size
			}
		}
		return io.MultiReader(frames...), nil
	}
	ref := &pb.ArtifactRef{}
	if len(parts) == 4 {
		ref.Namespace = proto.String(parts[2])
		ref.Name = proto.String(parts[3])
	}
	if target == pb.Artifact_Read_FullMethodName {
		request := &pb.ArtifactReadTarget{Artifact: ref}
		query, err := exactQuery(r.URL.RawQuery, "revision", "scope", "operation")
		if err != nil {
			fixtureUnknown(request)
		}
		if value, ok := query["revision"]; ok {
			ref.Revision = &value[0]
		}
		if value, ok := query["scope"]; ok {
			request.Scope = &value[0]
		}
		if value, ok := query["operation"]; ok {
			request.Operation = &value[0]
		}
		return bytes.NewReader(fixtureFrame(request)), nil
	}
	header := &pb.ArtifactWriteHeader{Artifact: ref, MediaType: proto.String(r.Header.Get("Content-Type"))}
	var body io.Reader = r.Body
	length := r.ContentLength
	if length == -1 {
		data, err := io.ReadAll(io.LimitReader(body, artifacts.MaxPayloadSize+1))
		if err != nil {
			return nil, err
		}
		body = bytes.NewReader(data)
		length = int64(len(data))
	}
	header.Size = proto.Uint64(uint64(length))
	expected, create, err := artifactWritePrecondition(r)
	if err == nil {
		if create {
			header.Precondition = &pb.ArtifactWriteHeader_CreateOnly{CreateOnly: true}
		} else {
			header.Precondition = &pb.ArtifactWriteHeader_ExpectedRevision{ExpectedRevision: *expected}
		}
	}
	if r.URL.RawQuery != "" {
		fixtureUnknown(header)
	}
	return io.MultiReader(bytes.NewReader(fixtureFrame(&pb.ArtifactWriteFrame{Frame: &pb.ArtifactWriteFrame_Header{Header: header}})), &payloadFixtureChunks{body: body, remaining: length}), nil
}

type payloadFixtureChunks struct {
	body      io.Reader
	remaining int64
	frame     *bytes.Reader
}

func (r *payloadFixtureChunks) Read(p []byte) (int, error) {
	if r.frame != nil && r.frame.Len() > 0 {
		return r.frame.Read(p)
	}
	if r.remaining <= 0 {
		return r.body.Read(p)
	}
	length := runtimewire.BulkChunkSize(int(r.remaining))
	chunk := make([]byte, length)
	if _, err := io.ReadFull(r.body, chunk); err != nil {
		return 0, err
	}
	r.remaining -= int64(length)
	r.frame = bytes.NewReader(fixtureFrame(&pb.ArtifactWriteFrame{Frame: &pb.ArtifactWriteFrame_Chunk{Chunk: chunk}}))
	return r.frame.Read(p)
}
func fixtureMessages(raw []byte) ([][]byte, error) {
	var values [][]byte
	for len(raw) > 0 {
		if len(raw) < 5 || raw[0] != 0 {
			return nil, fmt.Errorf("invalid native framing")
		}
		size := int(binary.BigEndian.Uint32(raw[1:5]))
		raw = raw[5:]
		if size > len(raw) {
			return nil, io.ErrUnexpectedEOF
		}
		values = append(values, raw[:size])
		raw = raw[size:]
	}
	return values, nil
}
func writePayloadFixtureResponse(w http.ResponseWriter, r *nativePayloadRecorder, original *http.Request, target string) {
	response := r.Result()
	statusText := response.Trailer.Get("Grpc-Status")
	if statusText == "" {
		statusText = r.Header().Get("Grpc-Status")
	}
	number, _ := strconv.Atoi(statusText)
	w.Header().Set("X-Request-ID", r.Header().Get("X-Request-ID"))
	if number != 0 {
		details := response.Trailer.Get("Grpc-Status-Details-Bin")
		if details == "" {
			details = r.Header().Get("Grpc-Status-Details-Bin")
		}
		raw, _ := base64.RawStdEncoding.DecodeString(strings.TrimRight(details, "="))
		native := new(rpcstatus.Status)
		_ = proto.Unmarshal(raw, native)
		detail := new(pb.ErrorDetail)
		for _, item := range native.Details {
			if item.MessageIs(detail) {
				_ = item.UnmarshalTo(detail)
			}
		}
		if detail.GetCode() == "" {
			detail.Code = proto.String("internal_error")
			detail.Retryable = proto.Bool(true)
		}
		status := 500
		switch codes.Code(number) {
		case codes.InvalidArgument:
			status = 400
		case codes.Unauthenticated:
			status = 401
		case codes.PermissionDenied:
			status = 403
		case codes.NotFound:
			status = 404
		case codes.FailedPrecondition:
			status = 409
		case codes.Aborted:
			status = 503
		case codes.Unavailable:
			status = 503
		case codes.ResourceExhausted:
			status = 503
		}
		if detail.GetCode() == "artifact_too_large" {
			status = 413
		}
		if detail.GetCode() == "artifact_history_maintenance_pending" {
			w.Header().Set("Retry-After", "1")
		}
		writeJSON(w, status, errorResponse{Code: detail.GetCode(), Message: "operation rejected", Retryable: detail.GetRetryable(), RequestID: detail.GetRequestId()})
		return
	}
	frames, err := fixtureMessages(r.Body.Bytes())
	if err != nil {
		panic(err)
	}
	if len(frames) == 0 {
		panic("native payload RPC returned no message")
	}
	if target == pb.Artifact_SubmitFinding_FullMethodName {
		reply := new(pb.FindingSubmissionReceipt)
		if err := proto.Unmarshal(frames[0], reply); err != nil {
			panic(err)
		}
		var value findingSubmissionFixture
		if err := runtimewire.DecodeDomain(reply, &value, runtimewire.BulkHeaderBytes, runtimewire.BulkFrameBytes); err != nil {
			panic(err)
		}
		status := 201
		if reply.GetReplayed() {
			status = 200
		}
		writeJSON(w, status, value)
		return
	}
	metadata := new(pb.ArtifactPayloadMetadata)
	var payload bytes.Buffer
	if target == pb.Artifact_Read_FullMethodName {
		for index, raw := range frames {
			frame := new(pb.ArtifactReadFrame)
			if err := proto.Unmarshal(raw, frame); err != nil {
				panic(err)
			}
			if index == 0 {
				metadata = frame.GetHeader()
			} else {
				payload.Write(frame.GetChunk())
			}
		}
	} else {
		if err := proto.Unmarshal(frames[0], metadata); err != nil {
			panic(err)
		}
	}
	w.Header().Set("ETag", strconv.Quote(metadata.GetArtifact().GetRevision()))
	w.Header().Set(bindingCreatedAtHeader, metadata.GetBindingCreatedAt())
	w.Header().Set(revisionCreatedAtHeader, metadata.GetRevisionCreatedAt())
	w.Header().Set("Cache-Control", "no-store")
	if target == pb.Artifact_Read_FullMethodName {
		w.Header().Set("Content-Type", metadata.GetMediaType())
		w.Header().Set("Content-Length", strconv.Itoa(payload.Len()))
		w.WriteHeader(200)
		_, _ = w.Write(payload.Bytes())
		return
	}
	result := artifactWriteFixture{APIVersion: metadata.GetApiVersion(), Artifact: contracts.ArtifactRef{Namespace: metadata.GetArtifact().GetNamespace(), Name: metadata.GetArtifact().GetName(), Revision: metadata.GetArtifact().Revision}, MediaType: metadata.GetMediaType(), Size: metadata.GetSize()}
	status := 200
	if original.Header.Get("If-None-Match") == "*" {
		status = 201
	}
	writeJSON(w, status, result)
}

type findingSubmissionFixture map[string]json.RawMessage

func exactQuery(raw string, allowed ...string) (url.Values, error) {
	return httpx.ExactQuery(errInvalidRequest, raw, allowed...)
}
func artifactWritePrecondition(r *http.Request) (expectedRevision *string, create bool, err error) {
	ifMatch := r.Header.Values("If-Match")
	ifNoneMatch := r.Header.Values("If-None-Match")
	if len(ifMatch) > 0 && len(ifNoneMatch) > 0 {
		return nil, false, fmt.Errorf("%w: If-Match and If-None-Match are mutually exclusive", errInvalidRequest)
	}
	if len(ifNoneMatch) > 0 {
		if len(ifNoneMatch) != 1 || strings.TrimSpace(ifNoneMatch[0]) != "*" {
			return nil, false, fmt.Errorf("%w: If-None-Match only supports *", errInvalidRequest)
		}
		return nil, true, nil
	}
	if len(ifMatch) != 1 {
		return nil, false, fmt.Errorf("%w: exactly one create/update precondition is required", errInvalidRequest)
	}
	revision, err := httpx.ParseStrongETag(ifMatch[0])
	switch {
	case errors.Is(err, httpx.ErrWeakETag):
		return nil, false, fmt.Errorf("%w: If-Match requires one strong revision ETag", errInvalidRequest)
	case err != nil:
		return nil, false, fmt.Errorf("%w: If-Match requires one quoted revision", errInvalidRequest)
	}
	return &revision, false, nil
}

func (h *handler) handleError(w http.ResponseWriter, err error) {
	failure := classifyPrivateError(err)
	if failure.code == "artifact_history_maintenance_pending" {
		w.Header().Set("Retry-After", strconv.Itoa(int(artifactpostgres.HistoryRetryAfter/time.Second)))
	}
	h.writeError(w, failure.Status(), failure.Code(), failure.message, failure.Retryable())
}

func (h *handler) writeError(w http.ResponseWriter, status int, code, message string, retryable bool) {
	writeJSON(w, status, errorResponse{
		Code: code, Message: message, Retryable: retryable,
		RequestID: requestid.FromResponse(w),
	})
}

func writeJSON(w http.ResponseWriter, status int, value any) {
	body, err := json.Marshal(value)
	if err != nil {
		http.Error(w, "internal error", http.StatusInternalServerError)
		return
	}
	body = append(body, '\n')
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Content-Length", strconv.Itoa(len(body)))
	w.Header().Set("Cache-Control", "no-store")
	w.WriteHeader(status)
	_, _ = w.Write(body)
}

// The former HTTP DTO is assertion vocabulary only; native receipts use
// ArtifactPayloadMetadata in production.
type artifactWriteFixture struct {
	APIVersion string                `json:"apiVersion"`
	Artifact   contracts.ArtifactRef `json:"artifact"`
	MediaType  string                `json:"mediaType"`
	Size       int64                 `json:"size"`
}
