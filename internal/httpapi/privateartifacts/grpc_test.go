package privateartifacts

import (
	"context"
	"crypto/x509"
	"encoding/json"
	"encoding/pem"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	pb "github.com/grauwolf32/contractor/api/v1alpha1/runtimepb"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"
	"github.com/grauwolf32/contractor/internal/localpki"
	"github.com/grauwolf32/contractor/internal/mtls"
	"github.com/grauwolf32/contractor/internal/runtimewire"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"
)

type allocationTestHarness struct {
	Artifact pb.ArtifactClient
	Client   pb.AllocationClient
	Context  context.Context
	HTTP     *http.Client
	URL      string
}

func allocationRPCFixture(t *testing.T, owner *handler) (pb.AllocationClient, context.Context) {
	fixture := newAllocationNativeHarness(t, owner)
	return fixture.Client, fixture.Context
}
func newAllocationNativeHarness(t *testing.T, owner *handler, wrappers ...func(http.Handler) http.Handler) allocationTestHarness {
	t.Helper()
	root := filepath.Join(t.TempDir(), "pki")
	generator := localpki.Generator{}
	ca, err := generator.InitCA(root, false)
	if err != nil {
		t.Fatal(err)
	}
	leaf := localpki.LeafOptions{DNSNames: []string{"localhost"}, IPAddresses: []net.IP{net.ParseIP("127.0.0.1")}}
	cp, err := generator.IssueControlPlane(root, localpki.ControlPlaneOptions{LeafOptions: leaf})
	if err != nil {
		t.Fatal(err)
	}
	agent, err := generator.IssueAgent(root, "rpc-agent", leaf)
	if err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(agent.Certificate)
	if err != nil {
		t.Fatal(err)
	}
	block, _ := pem.Decode(raw)
	cert, err := x509.ParseCertificate(block.Bytes)
	if err != nil {
		t.Fatal(err)
	}
	principal, err := mtls.RuntimeAgentID(cert)
	if err != nil {
		t.Fatal(err)
	}
	registry := owner.dependencies.Registry.(*fakeRegistry)
	registry.grant.RuntimeAgentID = principal
	// Preserve the injected domain adapter; production composition supplies the same owner.
	if owner.dependencies.Transfers == nil {
		owner.dependencies.Transfers = artifactpgtest.NewRuntime()
	}
	if owner.dependencies.Artifacts == nil {
		owner.dependencies.Artifacts = artifacts.NewService(newMemoryRepository())
	}
	native, err := newAllocationGRPCHandler(owner)
	if err != nil {
		t.Fatal(err)
	}
	dispatch := native
	for _, wrap := range wrappers {
		dispatch = wrap(dispatch)
	}
	server := httptest.NewUnstartedServer(dispatch)
	server.EnableHTTP2 = true
	server.Config.MaxHeaderBytes = runtimewire.MaxMetadataBytes
	server.TLS, err = mtls.ControlPlaneServerConfig(mtls.Files{Certificate: cp.Certificate, PrivateKey: cp.PrivateKey, CA: ca.Certificate})
	if err != nil {
		t.Fatal(err)
	}
	server.StartTLS()
	t.Cleanup(server.Close)
	clientTLS, err := mtls.ControlPlaneEndpointClientConfig(mtls.Files{Certificate: agent.Certificate, PrivateKey: agent.PrivateKey, CA: ca.Certificate})
	if err != nil {
		t.Fatal(err)
	}
	connection, err := grpc.NewClient(strings.TrimPrefix(server.URL, "https://"), grpc.WithTransportCredentials(credentials.NewTLS(clientTLS)), grpc.WithDisableRetry())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = connection.Close() })
	ctx, cancel := context.WithTimeout(t.Context(), 90*time.Second)
	t.Cleanup(cancel)
	ctx = metadata.AppendToOutgoingContext(ctx, strings.ToLower(RuntimeInstanceHeader), "runtime-1", strings.ToLower(allocationIDHeader), "allocation-1")
	httpTransport := &http.Transport{TLSClientConfig: clientTLS, ForceAttemptHTTP2: true}
	t.Cleanup(httpTransport.CloseIdleConnections)
	return allocationTestHarness{Artifact: pb.NewArtifactClient(connection), Client: pb.NewAllocationClient(connection), Context: ctx, HTTP: &http.Client{Transport: httpTransport}, URL: server.URL}
}

type unreadAllocationBody struct{}

func (unreadAllocationBody) Read([]byte) (int, error) { panic("rejected allocation RPC read a body") }
func (unreadAllocationBody) Close() error             { return nil }

func TestAllocationRPCRejectsAuthorityBeforeBody(t *testing.T) {
	owner := &handler{dependencies: Dependencies{Transfers: artifactpgtest.NewRuntime(), Registry: &fakeRegistry{grant: testGrant("run-a")}, Artifacts: artifacts.NewService(newMemoryRepository())}}
	native, err := newAllocationGRPCHandler(owner)
	if err != nil {
		t.Fatal(err)
	}
	for _, method := range []string{pb.Allocation_PlannerOperation_FullMethodName, pb.Allocation_GatewayRecovery_FullMethodName, pb.Allocation_ReadArtifactMetadata_FullMethodName, pb.Allocation_SubmitAuditEvidence_FullMethodName} {
		for _, test := range []struct {
			name, allocation, instance, want string
			trusted                          bool
		}{
			{name: "missing-mtls", allocation: "allocation-1", instance: "runtime-1", want: "16"},
			{name: "foreign-instance", allocation: "allocation-1", instance: "runtime-other", want: "9", trusted: true},
			{name: "foreign-allocation", allocation: "allocation-other", instance: "runtime-1", want: "9", trusted: true},
			{name: "missing-instance", allocation: "allocation-1", want: "3", trusted: true},
			{name: "missing-allocation", instance: "runtime-1", want: "3", trusted: true},
		} {
			t.Run(method+"/"+test.name, func(t *testing.T) {
				request := trustedRequest(http.MethodPost, method, unreadAllocationBody{})
				request.Header.Set(allocationIDHeader, test.allocation)
				request.Header.Set(RuntimeInstanceHeader, test.instance)
				if !test.trusted {
					request.TLS = nil
				}
				response := httptest.NewRecorder()
				native.ServeHTTP(response, request)
				want := test.want
				if want == "9" && method != pb.Allocation_PlannerOperation_FullMethodName && method != pb.Allocation_GatewayRecovery_FullMethodName {
					want = "5"
				}
				if response.Header().Get("Grpc-Status") != want {
					t.Fatalf("authority status=%v", response.Header())
				}
			})
		}
	}
}

// sitemapCommand sends a named native RPC. The recorder is only the existing
// domain assertion vocabulary; no HTTP command reaches production handlers.
func (h allocationTestHarness) sitemapCommand(t *testing.T, operation, key string, body any) *httptest.ResponseRecorder {
	t.Helper()
	raw, err := json.Marshal(body)
	if err != nil {
		t.Fatal(err)
	}
	var value map[string]json.RawMessage
	if err = json.Unmarshal(raw, &value); err != nil {
		t.Fatal(err)
	}
	encodedKey, _ := json.Marshal(key)
	value["idempotencyKey"] = encodedKey
	var input, output proto.Message
	switch operation {
	case "mutations":
		input = new(pb.SitemapMutationRequest)
	case "snapshots":
		input = new(pb.SitemapSnapshotRequest)
	case "observations":
		input = new(pb.SitemapCollectionRequest)
	default:
		t.Fatalf("unknown test operation %q", operation)
	}
	err = runtimewire.EncodeDomain(value, input, (256<<10)+rpcEnvelopeBytes, (256<<10)+rpcEnvelopeBytes)
	if err != nil {
		err = runtimewire.Error(h.Context, codes.InvalidArgument, "invalid_request", false)
	} else {
		switch request := input.(type) {
		case *pb.SitemapMutationRequest:
			output, err = h.Client.MutateSitemap(h.Context, request)
		case *pb.SitemapSnapshotRequest:
			output, err = h.Client.SnapshotSitemap(h.Context, request)
		case *pb.SitemapCollectionRequest:
			output, err = h.Client.CollectSitemap(h.Context, request)
		}
	}
	response := httptest.NewRecorder()
	if err != nil {
		current := status.Convert(err)
		details := current.Details()
		if len(details) != 1 {
			t.Fatalf("unclassified native error: %v", err)
		}
		detail := details[0].(*pb.ErrorDetail)
		code := 500
		switch current.Code() {
		case codes.InvalidArgument:
			code = 400
		case codes.PermissionDenied:
			code = 403
		case codes.NotFound:
			code = 404
		case codes.FailedPrecondition:
			code = 409
		case codes.Unavailable:
			code = 503
		}
		writeJSON(response, code, errorResponse{Code: detail.GetCode(), Message: "operation rejected", Retryable: detail.GetRetryable(), RequestID: detail.GetRequestId()})
	} else {
		var value map[string]json.RawMessage
		if err := runtimewire.DecodeDomain(output, &value, 16<<10, (256<<10)+rpcEnvelopeBytes); err != nil {
			t.Fatal(err)
		}
		writeJSON(response, 200, value)
	}
	return response
}
func (h allocationTestHarness) payloadRequest(t *testing.T, method, path string) *httptest.ResponseRecorder {
	t.Helper()
	parsed, parseErr := url.Parse(path)
	if parseErr != nil {
		t.Fatal(parseErr)
	}
	if strings.HasPrefix(parsed.Path, "/artifacts/") && (strings.HasSuffix(parsed.Path, "/access") || strings.HasSuffix(parsed.Path, "/metadata")) {
		return h.artifactRequest(t, path)
	}

	if method != http.MethodGet {
		t.Fatalf("payload fixture method=%s", method)
	}
	parts := strings.Split(strings.Trim(parsed.Path, "/"), "/")
	input := map[string]any{}
	ref := map[string]any{}
	if len(parts) == 3 && parts[0] == "artifacts" {
		ref["namespace"] = parts[1]
		ref["name"] = parts[2]
	}
	query, err := exactQuery(parsed.RawQuery, "revision", "scope", "operation")
	for key, values := range query {
		if key == "revision" {
			ref[key] = values[0]
		} else {
			input[key] = values[0]
		}
	}
	input["artifact"] = ref
	request := new(pb.ArtifactReadTarget)
	if err == nil {
		err = runtimewire.EncodeDomain(input, request, runtimewire.BulkHeaderBytes, runtimewire.BulkFrameBytes)
	}
	var frames []*pb.ArtifactReadFrame
	if err == nil {
		stream, callErr := h.Artifact.Read(h.Context, request)
		err = callErr
		if err == nil {
			for {
				frame, receiveErr := stream.Recv()
				if receiveErr == io.EOF {
					break
				}
				if receiveErr != nil {
					err = receiveErr
					break
				}
				frames = append(frames, frame)
			}
		}
	} else {
		err = runtimewire.Error(h.Context, codes.InvalidArgument, "invalid_request", false)
	}
	response := httptest.NewRecorder()
	if err != nil {
		return payloadFixtureFailure(t, err)
	}
	if len(frames) == 0 || frames[0].GetHeader() == nil {
		t.Fatal("native read omitted metadata")
	}
	header := frames[0].GetHeader()
	response.Header().Set("Content-Type", header.GetMediaType())
	response.Header().Set("ETag", strconv.Quote(header.GetArtifact().GetRevision()))
	response.Header().Set(bindingCreatedAtHeader, header.GetBindingCreatedAt())
	response.Header().Set(revisionCreatedAtHeader, header.GetRevisionCreatedAt())
	response.Header().Set("Cache-Control", "no-store")
	response.WriteHeader(200)
	for _, frame := range frames[1:] {
		_, _ = response.Write(frame.GetChunk())
	}
	return response
}
func payloadFixtureFailure(t *testing.T, err error) *httptest.ResponseRecorder {
	t.Helper()
	response := httptest.NewRecorder()
	current := status.Convert(err)
	var detail *pb.ErrorDetail
	for _, value := range current.Details() {
		if typed, ok := value.(*pb.ErrorDetail); ok {
			detail = typed
		}
	}
	if detail == nil {
		t.Fatal(err)
	}
	code := 500
	switch current.Code() {
	case codes.InvalidArgument:
		code = 400
	case codes.PermissionDenied:
		code = 403
	case codes.NotFound:
		code = 404
	case codes.FailedPrecondition:
		code = 409
	case codes.Aborted, codes.Unavailable, codes.ResourceExhausted:
		code = 503
	}
	writeJSON(response, code, errorResponse{Code: detail.GetCode(), Message: "operation rejected", Retryable: detail.GetRetryable(), RequestID: detail.GetRequestId()})
	return response
}

// artifactRequest translates the existing domain fixtures to named RPCs.
// Payload reads use the same native mTLS listener.
func (h allocationTestHarness) artifactRequest(t *testing.T, path string) *httptest.ResponseRecorder {
	t.Helper()
	path = strings.TrimPrefix(path, "/private/v1/allocations/allocation-1")
	parsed, err := url.Parse(path)
	if err != nil {
		t.Fatal(err)
	}
	parts := strings.Split(strings.Trim(parsed.Path, "/"), "/")
	var request, reply proto.Message
	input := map[string]any{}
	query, queryErr := url.ParseQuery(parsed.RawQuery)
	var encodeErr error
	if queryErr != nil {
		encodeErr = queryErr
	}
	if len(parts) == 1 && parts[0] == "artifacts" {
		request = new(pb.ArtifactListRequest)
		for key, values := range query {
			if len(values) != 1 {
				encodeErr = runtimewire.ErrInvalid
				continue
			}
			if key == "limit" {
				limit, parseErr := strconv.Atoi(values[0])
				if parseErr != nil {
					encodeErr = parseErr
				}
				input[key] = limit
			} else {
				input[key] = values[0]
			}
		}
	} else if len(parts) == 4 && parts[0] == "artifacts" && (parts[3] == "metadata" || parts[3] == "access") {
		request = new(pb.ArtifactReadTarget)
		ref := map[string]any{"namespace": parts[1], "name": parts[2]}
		for key, values := range query {
			if len(values) != 1 {
				encodeErr = runtimewire.ErrInvalid
				continue
			}
			if key == "revision" {
				ref[key] = values[0]
			} else {
				input[key] = values[0]
			}
		}
		input["artifact"] = ref
	} else {
		return h.payloadRequest(t, http.MethodGet, path)
	}
	if encodeErr == nil {
		encodeErr = runtimewire.EncodeDomain(input, request, 1<<20, (1<<20)+rpcEnvelopeBytes)
	}
	if encodeErr != nil {
		err = runtimewire.Error(h.Context, codes.InvalidArgument, "invalid_request", false)
	} else {
		switch value := request.(type) {
		case *pb.ArtifactListRequest:
			reply, err = h.Client.ListArtifacts(h.Context, value)
		case *pb.ArtifactReadTarget:
			if parts[3] == "access" {
				reply, err = h.Client.CheckArtifactAccess(h.Context, value)
			} else {
				reply, err = h.Client.ReadArtifactMetadata(h.Context, value)
			}
		}
	}
	response := httptest.NewRecorder()
	if err != nil {
		current := status.Convert(err)
		details := current.Details()
		if len(details) != 1 {
			t.Fatal(err)
		}
		detail := details[0].(*pb.ErrorDetail)
		statusCode := 500
		switch current.Code() {
		case codes.InvalidArgument:
			statusCode = 400
		case codes.PermissionDenied:
			statusCode = 403
		case codes.NotFound:
			statusCode = 404
		case codes.FailedPrecondition:
			statusCode = 409
		case codes.Unavailable:
			statusCode = 503
		}
		writeJSON(response, statusCode, errorResponse{Code: detail.GetCode(), Message: "operation rejected", Retryable: detail.GetRetryable(), RequestID: detail.GetRequestId()})
	} else if _, access := reply.(*pb.ArtifactAccessResponse); access {
		response.Header().Set("Cache-Control", "no-store")
		response.WriteHeader(204)
	} else {
		var value map[string]json.RawMessage
		if decodeErr := runtimewire.DecodeDomain(reply, &value, 1<<20, (1<<20)+rpcEnvelopeBytes); decodeErr != nil {
			t.Fatal(decodeErr)
		}
		writeJSON(response, 200, value)
	}
	return response
}
