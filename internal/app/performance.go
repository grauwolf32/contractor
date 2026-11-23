package app

import (
	"context"
	"io"
	"net/http"
	"time"

	"github.com/grauwolf32/contractor/internal/performance"
)

func newPerformanceDiagnostics(enabled bool, databaseURL string) *performance.Diagnostics {
	if !enabled {
		return nil
	}
	return performance.NewDiagnostics(performance.DiagnosticOptions{Open: func(ctx context.Context) (performance.DiagnosticBackend, error) {
		pool, err := performance.NewDiagnosticPool(ctx, databaseURL)
		if err != nil {
			return nil, err
		}
		return performance.NewDatabaseStore(pool), nil
	}})
}

// The disabled branch preserves original handler identity and never invokes
// the factory: no recorder, reader, buffer, clock or timer is created.
func instrumentPerformance(enabled bool, public, private http.Handler, factory func() *performance.Collector) (http.Handler, http.Handler, *performance.Collector) {
	if !enabled {
		return public, private, nil
	}
	collector := factory()
	return collector.Wrap(performance.Public, public), collector.Wrap(performance.Private, private), collector
}

// NewReadyHandler uses the working pool's context-aware Ping independently of
// performance collection. Earlier request deadlines always win.
func NewReadyHandler(check func(context.Context) error, public http.Handler) http.Handler {
	return newProcessHandler(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		ctx, cancel := context.WithTimeout(r.Context(), time.Second)
		defer cancel()
		w.Header().Set("Cache-Control", "no-store")
		if check == nil || check(ctx) != nil || ctx.Err() != nil {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusServiceUnavailable)
			_, _ = io.WriteString(w, "{\"status\":\"unavailable\"}\n")
			return
		}
		writeHealthy(w, r)
	}), public)
}
