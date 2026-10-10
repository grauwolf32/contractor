// Package reporting holds the execution telemetry a Runtime reports on the
// private wire: Worker execution reports with their metrics, tool-call
// records and errors; Runtime reports with adapter metrics, dropped spans and
// process resources; the allocation final report and its response; Worker
// completion diagnostics; the performance collection policy and request; and
// the live Worker State snapshot.
//
// It mirrors the Python runtime's contracts/reports.py; Worker State mirrors
// the state models of contracts/worker.py.
package reporting
