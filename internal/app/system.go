package app

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"time"
)

type backgroundRunner interface {
	Run(context.Context) error
}

type backgroundRunnerGroup []backgroundRunner

func (g backgroundRunnerGroup) Run(ctx context.Context) error {
	if len(g) == 0 {
		return errors.New("background runner group is empty")
	}
	groupContext, cancel := context.WithCancel(ctx)
	defer cancel()
	results := make(chan error, len(g))
	for _, runner := range g {
		go func(current backgroundRunner) { results <- current.Run(groupContext) }(runner)
	}
	first := <-results
	cancel()
	result := first
	for remaining := 1; remaining < len(g); remaining++ {
		result = errors.Join(result, <-results)
	}
	return result
}

// ServeSystem keeps the private mTLS API available while the public listener
// stops and the Scheduler drains. This lets in-flight allocation cleanup finish
// before Runtime Agent control and Artifact routes disappear.
func ServeSystem(
	ctx context.Context,
	publicListener net.Listener,
	privateListener net.Listener,
	shutdownTimeout time.Duration,
	logger *slog.Logger,
	publicHandler http.Handler,
	privateHandler http.Handler,
	runner backgroundRunner,
) error {
	if publicListener == nil || privateListener == nil || publicHandler == nil ||
		privateHandler == nil || runner == nil {
		return fmt.Errorf("Server system dependencies are incomplete")
	}
	if shutdownTimeout <= 0 {
		return fmt.Errorf("shutdown timeout must be positive")
	}
	baseContext := func(net.Listener) context.Context { return context.WithoutCancel(ctx) }
	publicServer := &http.Server{Handler: publicHandler, ReadHeaderTimeout: 5 * time.Second, BaseContext: baseContext}
	privateServer := &http.Server{Handler: privateHandler, ReadHeaderTimeout: 5 * time.Second, BaseContext: baseContext}
	schedulerContext, cancelScheduler := context.WithCancel(context.WithoutCancel(ctx))
	defer cancelScheduler()

	publicDone := make(chan error, 1)
	privateDone := make(chan error, 1)
	schedulerDone := make(chan error, 1)
	go func() { publicDone <- normalizeServeError(publicServer.Serve(publicListener)) }()
	go func() { privateDone <- normalizeServeError(privateServer.Serve(privateListener)) }()
	go func() { schedulerDone <- runner.Run(schedulerContext) }()

	logger.Info(
		"contractor server listening",
		"public_address", publicListener.Addr().String(),
	)

	var firstErr error
	publicFinished, privateFinished, schedulerFinished := false, false, false
	select {
	case <-ctx.Done():
	case err := <-publicDone:
		publicFinished = true
		firstErr = componentError("public HTTP server", err)
	case err := <-privateDone:
		privateFinished = true
		firstErr = componentError("private HTTP server", err)
	case err := <-schedulerDone:
		schedulerFinished = true
		firstErr = componentError("Workflow Scheduler", err)
	}

	shutdownContext, cancelShutdown := context.WithTimeout(context.Background(), shutdownTimeout)
	defer cancelShutdown()
	if err := publicServer.Shutdown(shutdownContext); err != nil {
		_ = publicServer.Close()
		firstErr = errors.Join(firstErr, fmt.Errorf("shutdown public HTTP server: %w", err))
	}
	if !publicFinished {
		select {
		case err := <-publicDone:
			firstErr = errors.Join(firstErr, componentError("public HTTP server", err))
		case <-shutdownContext.Done():
			firstErr = errors.Join(firstErr, fmt.Errorf("public HTTP server did not stop: %w", shutdownContext.Err()))
		}
	}

	cancelScheduler()
	if !schedulerFinished {
		select {
		case err := <-schedulerDone:
			firstErr = errors.Join(firstErr, componentError("Workflow Scheduler", err))
		case <-shutdownContext.Done():
			firstErr = errors.Join(firstErr, fmt.Errorf("Workflow Scheduler did not stop: %w", shutdownContext.Err()))
		}
	}

	if err := privateServer.Shutdown(shutdownContext); err != nil {
		_ = privateServer.Close()
		firstErr = errors.Join(firstErr, fmt.Errorf("shutdown private HTTP server: %w", err))
	}
	if !privateFinished {
		select {
		case err := <-privateDone:
			firstErr = errors.Join(firstErr, componentError("private HTTP server", err))
		case <-shutdownContext.Done():
			firstErr = errors.Join(firstErr, fmt.Errorf("private HTTP server did not stop: %w", shutdownContext.Err()))
		}
	}
	logger.Info("contractor server stopped")
	return firstErr
}

func normalizeServeError(err error) error {
	if errors.Is(err, http.ErrServerClosed) {
		return nil
	}
	return err
}

func componentError(component string, err error) error {
	if err == nil {
		return nil
	}
	return fmt.Errorf("%s: %w", component, err)
}
