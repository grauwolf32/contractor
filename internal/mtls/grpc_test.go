package mtls

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"github.com/grauwolf32/contractor/internal/mtlstest"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/wrapperspb"
)

func nativeChannelFixture(t *testing.T) (*RuntimeChannels, string, string, *atomic.Int64) {
	t.Helper()
	files := generateFiles(t, "native-channels")
	incoming, err := mtlstest.AgentServerConfig(mtlstest.Files(files.agent))
	if err != nil {
		t.Fatal(err)
	}
	outgoing, err := ControlPlaneEndpointClientConfig(files.controlPlane)
	if err != nil {
		t.Fatal(err)
	}
	principal, err := RuntimeAgentID(parseCertificate(t, files.agent.Certificate))
	if err != nil {
		t.Fatal(err)
	}
	owner, err := NewRuntimeChannels(outgoing)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(owner.Close)
	calls := new(atomic.Int64)
	server := grpc.NewServer(grpc.Creds(credentials.NewTLS(incoming)))
	server.RegisterService(&grpc.ServiceDesc{ServiceName: "test.Runtime", HandlerType: (*interface{})(nil), Methods: []grpc.MethodDesc{{MethodName: "Mutate", Handler: func(_ any, ctx context.Context, decode func(any) error, _ grpc.UnaryServerInterceptor) (any, error) {
		var value wrapperspb.StringValue
		if err := decode(&value); err != nil {
			return nil, err
		}
		calls.Add(1)
		if value.Value == "oversized-metadata" {
			_ = grpc.SetHeader(ctx, metadata.Pairs("oversized", strings.Repeat("x", privateRPCMetadataBytes)))
		}
		if value.Value == "lost-reply" {
			return nil, status.Error(codes.Unavailable, "reply unavailable")
		}
		return &value, nil
	}}}}, struct{}{})
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	done := make(chan error, 1)
	go func() { done <- server.Serve(listener) }()
	t.Cleanup(func() { owner.Close(); server.Stop(); <-done; _ = listener.Close() })
	return owner, "https://" + listener.Addr().String(), principal, calls
}
func invokeNative(t *testing.T, channel *grpc.ClientConn, value string) error {
	t.Helper()
	ctx, cancel := context.WithTimeout(t.Context(), 2*time.Second)
	defer cancel()
	return channel.Invoke(ctx, "/test.Runtime/Mutate", wrapperspb.String(value), &wrapperspb.StringValue{})
}
func TestRuntimeChannelsRetireWithoutInterruptingSiblingAllocations(t *testing.T) {
	owner, origin, principal, _ := nativeChannelFixture(t)
	now := time.Now()
	owner.now = func() time.Time { return now }
	first, releaseFirst, err := owner.Borrow(t.Context(), origin, principal)
	if err != nil {
		t.Fatal(err)
	}
	second, releaseSecond, err := owner.Borrow(t.Context(), origin, principal)
	if err != nil {
		t.Fatal(err)
	}
	if first != second {
		t.Fatal("concurrent allocations did not reuse channel")
	}
	if err := invokeNative(t, first, "first"); err != nil {
		t.Fatal(err)
	}
	owner.Evict(principal)
	fresh, releaseFresh, err := owner.Borrow(t.Context(), origin, principal)
	if err != nil {
		t.Fatal(err)
	}
	if fresh == first {
		t.Fatal("released allocation's generation was reused")
	}
	releaseFirst()
	if err := invokeNative(t, second, "sibling"); err != nil {
		t.Fatal("release killed sibling:", err)
	}
	releaseSecond()
	if err := invokeNative(t, first, "retired"); err == nil {
		t.Fatal("last borrower left retired channel open")
	}
	releaseFresh()
	now = now.Add(privateConnectionReuseLifetime)
	rotated, releaseRotated, err := owner.Borrow(t.Context(), origin, principal)
	if err != nil {
		t.Fatal(err)
	}
	if rotated == fresh {
		t.Fatal("aged channel reused")
	}
	releaseRotated()
	owner.Close()
	if _, _, err := owner.Borrow(t.Context(), origin, principal); err == nil {
		t.Fatal("closed owner accepted call")
	}
}
func TestRuntimeChannelsBoundAdmissionAndNeverReplayMutation(t *testing.T) {
	owner, origin, principal, calls := nativeChannelFixture(t)
	releases := []func(){}
	for range privateConnectionsPerPeer {
		_, release, err := owner.Borrow(t.Context(), origin, principal)
		if err != nil {
			t.Fatal(err)
		}
		releases = append(releases, release)
	}
	if _, _, err := owner.Borrow(t.Context(), origin, principal); err == nil {
		t.Fatal("unbounded RPC admission")
	}
	for _, release := range releases {
		release()
	}
	channel, release, err := owner.Borrow(t.Context(), origin, principal)
	if err != nil {
		t.Fatal(err)
	}
	defer release()
	if err := invokeNative(t, channel, "lost-reply"); status.Code(err) != codes.Unavailable {
		t.Fatalf("status = %v", err)
	}
	if calls.Load() != 1 {
		t.Fatalf("mutation replayed %d times", calls.Load())
	}
	wrong, releaseWrong, err := owner.Borrow(t.Context(), origin, strings.Repeat("0", 64))
	if err != nil {
		t.Fatal(err)
	}
	defer releaseWrong()
	if err := invokeNative(t, wrong, "forbidden"); err == nil {
		t.Fatal("wrong principal reached RPC")
	}
	if !owner.PeerRejected(wrong) {
		t.Fatal("peer rejection was classified as transient unavailability")
	}
	if calls.Load() != 1 {
		t.Fatal("wrong principal dispatched a request")
	}
}

func TestRuntimeChannelsCloseIdleAndEvictLeastRecentlyUsed(t *testing.T) {
	owner, origin, principal, _ := nativeChannelFixture(t)
	now := time.Now()
	owner.now = func() time.Time { return now }
	var expire func()
	owner.afterFunc = func(_ time.Duration, callback func()) *time.Timer { expire = callback; return nil }
	first, release, err := owner.Borrow(t.Context(), origin, principal)
	if err != nil {
		t.Fatal(err)
	}
	if err := invokeNative(t, first, "before-idle"); err != nil {
		t.Fatal(err)
	}
	release()
	now = now.Add(privateConnectionIdleReuse)
	expire()
	if err := invokeNative(t, first, "closed-idle"); err == nil {
		t.Fatal("idle timer left channel open")
	}
	for i := range runtimeAgentTransportCapacity {
		_, release, err := owner.Borrow(t.Context(), fmt.Sprintf("https://runtime-%d.example", i), principal)
		if err != nil {
			t.Fatal(err)
		}
		release()
		now = now.Add(time.Nanosecond)
	}
	_, release, err = owner.Borrow(t.Context(), origin, principal)
	if err != nil {
		t.Fatal("idle capacity was not evicted:", err)
	}
	release()
	if len(owner.entries) != runtimeAgentTransportCapacity {
		t.Fatal("capacity not bounded")
	}
	if _, exists := owner.entries[principal+"@runtime-0.example:443"]; exists {
		t.Fatal("oldest idle endpoint retained")
	}
}

func TestSharedRuntimeChannelMetadataBound(t *testing.T) {
	var limits struct {
		MetadataBytes uint32 `json:"metadataBytes"`
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "v1alpha1", "grpc-limits.json"), &limits); err != nil {
		t.Fatal(err)
	}
	if limits.MetadataBytes != privateRPCMetadataBytes {
		t.Fatal("metadata bound differs from wire contract")
	}
	owner, origin, principal, calls := nativeChannelFixture(t)
	channel, release, err := owner.Borrow(t.Context(), origin, principal)
	if err != nil {
		t.Fatal(err)
	}
	defer release()
	if err := invokeNative(t, channel, "ordinary"); err != nil {
		t.Fatal(err)
	}
	if err := invokeNative(t, channel, "oversized-metadata"); err == nil {
		t.Fatal("unbounded response metadata accepted")
	}
	if calls.Load() != 2 {
		t.Fatal("metadata failure replayed mutation")
	}
}
