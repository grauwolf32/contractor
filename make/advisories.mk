# Live advisory scans. Their verdict changes whenever upstream publishes an
# advisory, without any code change, so CI runs them in a separate advisory
# job instead of in the deterministic release-verify gate.

.PHONY: advisories test-go-vulnerabilities test-runtime-dependencies-audit

advisories: test-go-vulnerabilities test-runtime-dependencies-audit

test-go-vulnerabilities:
	go run golang.org/x/vuln/cmd/govulncheck@v1.8.0 ./cmd/...

# Audits exactly the locked production Runtime graph with a hash-pinned
# scanner, after checking that the scanner reports a known-vulnerable pin.
test-runtime-dependencies-audit:
	python3 scripts/audit_runtime_dependencies.py
