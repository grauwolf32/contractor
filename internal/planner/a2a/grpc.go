package a2a

import (
	"context"
	"time"

	sdk "github.com/a2aproject/a2a-go/v2/a2a"
	"github.com/a2aproject/a2a-go/v2/a2aclient"
	a2agrpc "github.com/a2aproject/a2a-go/v2/a2agrpc/v1"
	a2apb "github.com/a2aproject/a2a-go/v2/a2apb/v1"
	"github.com/grauwolf32/contractor/internal/helpers/requestid"
	"github.com/grauwolf32/contractor/internal/mtls"
	"github.com/grauwolf32/contractor/internal/runtimewire"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
)

func sdkClientBuilder(channels *mtls.RuntimeChannels, principal string, timeout time.Duration) clientBuilder {
	return func(ctx context.Context, card *sdk.AgentCard) (protocolClient, error) {
		return a2aclient.NewFactory(a2aclient.WithDefaultsDisabled(), a2aclient.WithTransport(sdk.TransportProtocolGRPC, a2aclient.TransportFactoryFn(func(_ context.Context, _ *sdk.AgentCard, iface *sdk.AgentInterface) (a2aclient.Transport, error) {
			conn := &invocationConnection{channels: channels, origin: "https://" + iface.URL, principal: principal, timeout: timeout}
			return a2agrpc.NewGRPCTransportFromClient(a2apb.NewA2AServiceClient(conn)), nil
		})), a2aclient.WithConfig(a2aclient.Config{AcceptedOutputModes: []string{workerCompletionMediaType}, PreferredTransports: []sdk.TransportProtocol{sdk.TransportProtocolGRPC}})).CreateFromCard(ctx, card)
	}
}

type invocationConnection struct {
	channels          *mtls.RuntimeChannels
	origin, principal string
	timeout           time.Duration
}

func (c *invocationConnection) Invoke(ctx context.Context, method string, input, output any, options ...grpc.CallOption) error {
	ctx, cancel := context.WithTimeout(ctx, c.timeout)
	defer cancel()
	conn, release, err := c.channels.Borrow(ctx, c.origin, c.principal)
	if err != nil {
		return err
	}
	defer release()
	ctx = metadata.AppendToOutgoingContext(ctx, "x-request-id", requestid.Ensure(ctx))
	options = append(options, grpc.MaxCallRecvMsgSize(runtimewire.MaxA2ABytes), grpc.MaxCallSendMsgSize(runtimewire.MaxA2ABytes))
	err = conn.Invoke(ctx, method, input, output, options...)
	if err != nil && c.channels.PeerRejected(conn) {
		c.channels.Evict(c.principal)
		return status.Error(codes.Unauthenticated, "invalid_worker_handle")
	}
	// The official SDK maps an unclassified UNAVAILABLE to INTERNAL and loses
	// its retry class. Preserve only this safe transport class before mapping.
	if status.Code(err) == codes.Unavailable {
		return retryableRPCFailure{}
	}
	return err
}
func (*invocationConnection) NewStream(context.Context, *grpc.StreamDesc, string, ...grpc.CallOption) (grpc.ClientStream, error) {
	return nil, status.Error(codes.Unimplemented, "Worker streaming is not enabled")
}

type retryableRPCFailure struct{}

func (retryableRPCFailure) Error() string   { return "A2A transport unavailable" }
func (retryableRPCFailure) Timeout() bool   { return false }
func (retryableRPCFailure) Temporary() bool { return true }
