# Go profiling

Contractor Server can expose a dedicated, loopback-only Go diagnostics listener.
It is disabled by default and is independent from Operations performance metrics:

```sh
contractor-server serve \
  --performance-metrics=false \
  --pprof=true \
  --pprof-listen=127.0.0.1:6060
```

The equivalent environment settings are `CONTRACTOR_PPROF=true` and
`CONTRACTOR_PPROF_LISTEN=127.0.0.1:6060`. Command-line values take precedence.
Changing either setting requires a Server restart. Enabling the listener alone
does not start CPU or execution-trace recording; those begin only while a bounded
request is active.

The listener accepts numeric loopback addresses only. It is separate from the
public and private application listeners and exposes exactly:

- `/debug/pprof/`
- `/debug/pprof/profile`
- `/debug/pprof/heap`
- `/debug/pprof/allocs`
- `/debug/pprof/goroutine`
- `/debug/pprof/threadcreate`
- `/debug/pprof/trace`
- `/debug/pprof/symbol`

In particular, `cmdline`, block, and mutex profiles are not exposed. Profile
bytes are streamed to the diagnostic client and are not stored in PostgreSQL,
execution telemetry, or Operations history.

CPU capture defaults to 30 seconds and accepts integer `seconds` from 1 through
60. Trace capture defaults to 1 second and accepts 1 through 10. Delta requests
for heap, allocs, goroutine, and threadcreate accept 1 through 60. Duplicate,
unknown, malformed, or excessive query parameters are rejected. Only one CPU or
trace capture and one additional snapshot or delta request may be active at the
same time; excess requests receive HTTP 409 and are not queued.

Typical local commands are:

```sh
go tool pprof 'http://127.0.0.1:6060/debug/pprof/profile?seconds=30'
go tool pprof 'http://127.0.0.1:6060/debug/pprof/heap?gc=1'
curl -fsS -o contractor.trace \
  'http://127.0.0.1:6060/debug/pprof/trace?seconds=5'
go tool trace contractor.trace
```

For a remote VM, keep Contractor bound to loopback and use an SSH tunnel:

```sh
ssh -N -L 6060:127.0.0.1:6060 operator@contractor-vm
go tool pprof 'http://127.0.0.1:6060/debug/pprof/profile?seconds=30'
```

Treat profiles as sensitive operational data even though the listener omits
`cmdline`. Do not publish the tunnel or bind the diagnostics listener to a LAN or
wildcard address. Disconnecting the client or stopping Server cancels an active
recording and releases its capacity slot within the configured shutdown bound.
