package app

import (
	"fmt"

	runtimeprofiling "github.com/grauwolf32/contractor/internal/profiling"
)

func configureProfiling(cfg Config) (*runtimeprofiling.Server, error) {
	if !cfg.Pprof {
		return nil, nil
	}
	server, err := runtimeprofiling.Listen(runtimeprofiling.Options{
		ListenAddress: cfg.PprofListen, ShutdownTimeout: cfg.ShutdownTimeout,
	})
	if err != nil {
		return nil, fmt.Errorf("configure Go profiling: %w", err)
	}
	return server, nil
}
