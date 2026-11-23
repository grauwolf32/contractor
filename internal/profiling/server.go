package profiling

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"strconv"
	"time"
)

type Options struct {
	ListenAddress   string
	ShutdownTimeout time.Duration
}

// Server owns the already-bound diagnostic listener and its explicit handler.
// Binding is separate from Run so an enabled startup fails before serving any
// application endpoint when the requested address is unavailable.
type Server struct {
	listener        net.Listener
	httpServer      *http.Server
	shutdownTimeout time.Duration
	cancelRequests  context.CancelFunc
}

func Listen(options Options) (*Server, error) {
	if !validListenAddress(options.ListenAddress) {
		return nil, errors.New("profiling listen address must be a numeric loopback IP and non-zero TCP port")
	}
	listener, err := net.Listen("tcp", options.ListenAddress)
	if err != nil {
		return nil, errors.New("profiling listener is unavailable")
	}
	server, err := NewWithListener(listener, options.ShutdownTimeout)
	if err != nil {
		_ = listener.Close()
		return nil, err
	}
	return server, nil
}

// NewWithListener exists for process tests that need an OS-assigned loopback
// port. Production startup uses Listen and therefore cannot request port zero.
func NewWithListener(listener net.Listener, shutdownTimeout time.Duration) (*Server, error) {
	if listener == nil || shutdownTimeout <= 0 {
		return nil, errors.New("profiling server options are invalid")
	}
	address, ok := listener.Addr().(*net.TCPAddr)
	if !ok || address.IP == nil || !address.IP.IsLoopback() || address.Port < 1 || address.Port > 65535 {
		return nil, errors.New("profiling listener must be loopback TCP")
	}
	requestContext, cancelRequests := context.WithCancel(context.Background())
	return &Server{
		listener: listener,
		httpServer: &http.Server{
			Handler:           NewHandler(),
			ReadHeaderTimeout: 5 * time.Second,
			MaxHeaderBytes:    64 * 1024,
			BaseContext: func(net.Listener) context.Context {
				return requestContext
			},
		},
		shutdownTimeout: shutdownTimeout,
		cancelRequests:  cancelRequests,
	}, nil
}

func (s *Server) Addr() net.Addr { return s.listener.Addr() }

func (s *Server) Run(ctx context.Context) error {
	done := make(chan error, 1)
	go func() { done <- s.httpServer.Serve(s.listener) }()
	select {
	case err := <-done:
		if ctx.Err() != nil && isClosedServerError(err) {
			return nil
		}
		if isClosedServerError(err) {
			return errors.New("profiling listener stopped unexpectedly")
		}
		return fmt.Errorf("serve profiling listener: %w", err)
	case <-ctx.Done():
	}

	s.cancelRequests()
	shutdownContext, cancel := context.WithTimeout(context.Background(), s.shutdownTimeout)
	defer cancel()
	shutdownErr := s.httpServer.Shutdown(shutdownContext)
	if shutdownErr != nil {
		_ = s.httpServer.Close()
	}
	serveErr := <-done
	if shutdownErr != nil {
		return fmt.Errorf("shutdown profiling listener: %w", shutdownErr)
	}
	if !isClosedServerError(serveErr) {
		return fmt.Errorf("serve profiling listener during shutdown: %w", serveErr)
	}
	return nil
}

func (s *Server) Close() error {
	s.cancelRequests()
	serverErr := s.httpServer.Close()
	listenerErr := s.listener.Close()
	if isClosedServerError(serverErr) {
		serverErr = nil
	}
	if errors.Is(listenerErr, net.ErrClosed) {
		listenerErr = nil
	}
	return errors.Join(serverErr, listenerErr)
}

func validListenAddress(value string) bool {
	host, port, err := net.SplitHostPort(value)
	if err != nil || port == "" {
		return false
	}
	address := net.ParseIP(host)
	if address == nil || !address.IsLoopback() {
		return false
	}
	for _, character := range port {
		if character < '0' || character > '9' {
			return false
		}
	}
	number, err := strconv.Atoi(port)
	return err == nil && number >= 1 && number <= 65535
}

func isClosedServerError(err error) bool {
	return err == nil || errors.Is(err, http.ErrServerClosed) || errors.Is(err, net.ErrClosed)
}
