package runtimeconfig

import (
	"bytes"
	"context"
	"crypto/sha256"
	"crypto/x509"
	"encoding/hex"
	"encoding/json"
	"encoding/pem"
	"errors"
	"fmt"
	"io"
	"net/url"
	"regexp"
	"sort"
	"strings"
	"unicode/utf8"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/ucarion/jcs"
)

const (
	maxDocumentBytes = 128 * 1024
	maxURLBytes      = 2048
	maxPEMBytes      = 64 * 1024
)

var (
	idPattern      = regexp.MustCompile(`^[a-z][a-z0-9_-]*$`)
	versionPattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._+-]*$`)
	digestPattern  = regexp.MustCompile(`^sha256:[0-9a-f]{64}$`)
)

type optional[T any] struct {
	present bool
	null    bool
	value   T
}

func (o *optional[T]) UnmarshalJSON(data []byte) error {
	o.present = true
	if bytes.Equal(bytes.TrimSpace(data), []byte("null")) {
		o.null = true
		return nil
	}
	return decodeStrict(data, &o.value)
}

type documentSource struct {
	APIVersion string         `json:"apiVersion"`
	Kind       string         `json:"kind"`
	Metadata   metadataSource `json:"metadata"`
	Spec       specSource     `json:"spec"`
}

type metadataSource struct {
	Name    string `json:"name"`
	Version string `json:"version"`
}

type specSource struct {
	Worker  optional[workerSource]  `json:"worker"`
	Planner optional[plannerSource] `json:"planner"`
}

type workerSource struct {
	LLMGateway optional[llmGatewaySource] `json:"llmGateway"`
	Telemetry  optional[telemetrySource]  `json:"telemetry"`
	HTTPProxy  optional[httpProxySource]  `json:"httpProxy"`
	Caido      optional[caidoSource]      `json:"caido"`
}

type plannerSource struct {
	Telemetry optional[telemetrySource] `json:"telemetry"`
}

type llmGatewaySource struct {
	Gateway    optional[json.RawMessage] `json:"gateway"`
	Credential optional[string]          `json:"credential"`
}

type telemetrySource struct {
	Adapter             string           `json:"adapter"`
	Endpoint            string           `json:"endpoint"`
	Credential          optional[string] `json:"credential"`
	CaptureContent      optional[bool]   `json:"captureContent"`
	FlushTimeoutSeconds optional[int]    `json:"flushTimeoutSeconds"`
}

type httpProxySource struct {
	Adapter     string           `json:"adapter"`
	ProxyURL    string           `json:"proxyUrl"`
	Credential  optional[string] `json:"credential"`
	CABundlePEM optional[string] `json:"caBundlePem"`
	Targets     []string         `json:"targets"`
}

type caidoSource struct {
	Adapter               string           `json:"adapter"`
	Endpoint              string           `json:"endpoint"`
	Credential            optional[string] `json:"credential"`
	CABundlePEM           optional[string] `json:"caBundlePem"`
	RequestTimeoutSeconds optional[int]    `json:"requestTimeoutSeconds"`
}

type GatewayResolver interface {
	ResolveLLMGateway(context.Context, string) (contracts.ResolvedLLMGatewayConfig, error)
}

type GatewayResolverFunc func(context.Context, string) (contracts.ResolvedLLMGatewayConfig, error)

func (f GatewayResolverFunc) ResolveLLMGateway(ctx context.Context, selector string) (contracts.ResolvedLLMGatewayConfig, error) {
	return f(ctx, selector)
}

type PreparedPublication struct {
	name             string
	version          string
	source           documentSource
	authorCanonical  []byte
	requestDigest    string
	gatewaySelectors map[string]string
}

func (p PreparedPublication) Name() string          { return p.name }
func (p PreparedPublication) Version() string       { return p.version }
func (p PreparedPublication) RequestDigest() string { return p.requestDigest }

// PreparePublication validates and normalizes an author request without
// consulting mutable configuration catalogs. This separation is what lets a
// service perform durable idempotency replay before Gateway resolution.
func PreparePublication(data []byte) (PreparedPublication, error) {
	if len(data) == 0 || len(data) > maxDocumentBytes || !utf8.Valid(data) {
		return PreparedPublication{}, invalid("document must be non-empty UTF-8 and at most 128 KiB")
	}
	if err := rejectInvalidUnicodeEscapes(data); err != nil {
		return PreparedPublication{}, err
	}
	if err := rejectDuplicateKeys(data); err != nil {
		return PreparedPublication{}, err
	}
	var source documentSource
	if err := decodeStrict(data, &source); err != nil {
		return PreparedPublication{}, invalid("document does not match the RuntimeConfig schema")
	}
	if source.APIVersion != APIVersion || source.Kind != Kind {
		return PreparedPublication{}, invalid("apiVersion and kind must be %q and %q", APIVersion, Kind)
	}
	if err := validateID("metadata.name", source.Metadata.Name, 63); err != nil {
		return PreparedPublication{}, err
	}
	if !versionPattern.MatchString(source.Metadata.Version) || len(source.Metadata.Version) > 128 {
		return PreparedPublication{}, invalid("metadata.version is invalid")
	}
	if source.Metadata.Name == BuiltInName && source.Metadata.Version == BuiltInVersion {
		return PreparedPublication{}, fmt.Errorf("%w: built-in identity cannot be published", ErrReserved)
	}
	selectors, canonicalSpec, operations, err := validateSource(source.Spec, true)
	if err != nil {
		return PreparedPublication{}, err
	}
	if operations == 0 {
		return PreparedPublication{}, invalid("spec must contain at least one patch operation")
	}
	canonical, err := canonicalize(map[string]any{
		"apiVersion": APIVersion,
		"kind":       Kind,
		"metadata":   map[string]any{"name": source.Metadata.Name, "version": source.Metadata.Version},
		"spec":       canonicalSpec,
	})
	if err != nil {
		return PreparedPublication{}, invalid("canonicalize author request")
	}
	if len(canonical) > maxDocumentBytes {
		return PreparedPublication{}, invalid("normalized document exceeds 128 KiB")
	}
	return PreparedPublication{
		name: source.Metadata.Name, version: source.Metadata.Version, source: source,
		authorCanonical: canonical, requestDigest: digest(canonical), gatewaySelectors: selectors,
	}, nil
}

func (p PreparedPublication) Resolve(ctx context.Context, resolver GatewayResolver) (Version, error) {
	if p.name == "" || len(p.authorCanonical) == 0 {
		return Version{}, invalid("publication was not prepared")
	}
	resolved := make(map[string]contracts.LLMGatewayConfigRef, len(p.gatewaySelectors))
	paths := make([]string, 0, len(p.gatewaySelectors))
	for path := range p.gatewaySelectors {
		paths = append(paths, path)
	}
	sort.Strings(paths)
	for _, path := range paths {
		selector := p.gatewaySelectors[path]
		if resolver == nil {
			return Version{}, invalid("Gateway resolver is required")
		}
		gateway, err := resolver.ResolveLLMGateway(ctx, selector)
		if err != nil {
			return Version{}, invalid("resolve %s", path)
		}
		if err := gateway.Validate(); err != nil || gateway.Ref.GatewayID+"@"+gateway.Ref.Version != selector {
			return Version{}, invalid("resolved %s does not match its exact selector", path)
		}
		resolved[path] = gateway.Ref
	}
	spec, canonicalSpec, _, err := materializeSource(p.source.Spec, false, resolved)
	if err != nil {
		return Version{}, err
	}
	canonical, err := canonicalize(map[string]any{
		"apiVersion": APIVersion,
		"kind":       Kind,
		"metadata":   map[string]any{"name": p.name, "version": p.version},
		"spec":       canonicalSpec,
	})
	if err != nil || len(canonical) > maxDocumentBytes {
		return Version{}, invalid("normalized immutable document is invalid")
	}
	return Version{Ref: Ref{Name: p.name, Version: p.version, Digest: digest(canonical)}, Spec: spec, CanonicalDocument: canonical}, nil
}

func DecodeStoredDocument(data []byte) (Version, error) {
	if len(data) == 0 || len(data) > maxDocumentBytes || !utf8.Valid(data) {
		return Version{}, invalid("stored document has invalid size or encoding")
	}
	if err := rejectInvalidUnicodeEscapes(data); err != nil {
		return Version{}, err
	}
	if err := rejectDuplicateKeys(data); err != nil {
		return Version{}, err
	}
	var source documentSource
	if err := decodeStrict(data, &source); err != nil {
		return Version{}, invalid("stored document does not match the RuntimeConfig schema")
	}
	if source.APIVersion != APIVersion || source.Kind != Kind {
		return Version{}, invalid("stored document envelope is invalid")
	}
	if err := validateID("metadata.name", source.Metadata.Name, 63); err != nil {
		return Version{}, err
	}
	if !versionPattern.MatchString(source.Metadata.Version) || len(source.Metadata.Version) > 128 {
		return Version{}, invalid("stored metadata.version is invalid")
	}
	spec, canonicalSpec, operations, err := materializeSource(source.Spec, false, nil)
	if err != nil {
		return Version{}, err
	}
	if operations == 0 && !(source.Metadata.Name == BuiltInName && source.Metadata.Version == BuiltInVersion) {
		return Version{}, invalid("only the built-in RuntimeConfig may be empty")
	}
	canonical, err := canonicalize(map[string]any{
		"apiVersion": APIVersion, "kind": Kind,
		"metadata": map[string]any{"name": source.Metadata.Name, "version": source.Metadata.Version},
		"spec":     canonicalSpec,
	})
	if err != nil || !bytes.Equal(canonical, data) {
		return Version{}, invalid("stored document is not exact normalized JCS")
	}
	return Version{Ref: Ref{Name: source.Metadata.Name, Version: source.Metadata.Version, Digest: digest(canonical)}, Spec: spec, CanonicalDocument: canonical, BuiltIn: operations == 0}, nil
}

func validateSource(source specSource, author bool) (map[string]string, map[string]any, int, error) {
	spec, canonical, operations, err := materializeSource(source, author, nil)
	_ = spec
	selectors := make(map[string]string)
	if err == nil {
		collectGatewaySelectors(source, selectors)
	}
	return selectors, canonical, operations, err
}

func materializeSource(source specSource, author bool, resolved map[string]contracts.LLMGatewayConfigRef) (Spec, map[string]any, int, error) {
	var result Spec
	canonical := make(map[string]any)
	operations := 0
	if source.Worker.present {
		if source.Worker.null {
			return Spec{}, nil, 0, invalid("spec.worker cannot be null")
		}
		worker, workerMap, count, err := materializeWorker(source.Worker.value, author, resolved)
		if err != nil {
			return Spec{}, nil, 0, err
		}
		if len(workerMap) == 0 {
			return Spec{}, nil, 0, invalid("spec.worker cannot be empty")
		}
		result.Worker = worker
		canonical["worker"] = workerMap
		operations += count
	}
	if source.Planner.present {
		if source.Planner.null {
			return Spec{}, nil, 0, invalid("spec.planner cannot be null")
		}
		planner := make(map[string]any)
		if source.Planner.value.Telemetry.present {
			patch, value, err := materializeTelemetry("spec.planner.telemetry", source.Planner.value.Telemetry)
			if err != nil {
				return Spec{}, nil, 0, err
			}
			result.Planner.Telemetry = patch
			planner["telemetry"] = value
			operations++
		}
		if len(planner) == 0 {
			return Spec{}, nil, 0, invalid("spec.planner cannot be empty")
		}
		canonical["planner"] = planner
	}
	return result, canonical, operations, nil
}

func materializeWorker(source workerSource, author bool, resolved map[string]contracts.LLMGatewayConfigRef) (WorkerPatch, map[string]any, int, error) {
	var result WorkerPatch
	canonical := make(map[string]any)
	operations := 0
	if source.LLMGateway.present {
		if source.LLMGateway.null {
			return WorkerPatch{}, nil, 0, invalid("spec.worker.llmGateway cannot be null")
		}
		patch := LLMGatewayPatch{Present: true}
		value := make(map[string]any)
		if source.LLMGateway.value.Gateway.present {
			if source.LLMGateway.value.Gateway.null {
				return WorkerPatch{}, nil, 0, invalid("spec.worker.llmGateway.gateway cannot be null")
			}
			path := "spec.worker.llmGateway.gateway"
			if author {
				var selector string
				if err := json.Unmarshal(source.LLMGateway.value.Gateway.value, &selector); err != nil || validateSelector(selector) != nil {
					return WorkerPatch{}, nil, 0, invalid("%s must be an exact selector string", path)
				}
				value["gateway"] = selector
			} else {
				var ref contracts.LLMGatewayConfigRef
				if resolved != nil {
					ref = resolved[path]
				} else if err := decodeStrict(source.LLMGateway.value.Gateway.value, &ref); err != nil {
					return WorkerPatch{}, nil, 0, invalid("%s must be an exact Gateway ref", path)
				}
				if err := ref.ValidateRef(); err != nil {
					return WorkerPatch{}, nil, 0, invalid("%s is invalid", path)
				}
				patch.Gateway = Field[contracts.LLMGatewayConfigRef]{Present: true, Value: ref}
				value["gateway"] = map[string]any{"gatewayId": ref.GatewayID, "version": ref.Version, "digest": ref.Digest}
			}
			operations++
		}
		if source.LLMGateway.value.Credential.present {
			patch.Credential.Present = true
			if source.LLMGateway.value.Credential.null {
				patch.Credential.Clear = true
				value["credential"] = nil
			} else {
				if err := validateID("spec.worker.llmGateway.credential", source.LLMGateway.value.Credential.value, 128); err != nil {
					return WorkerPatch{}, nil, 0, err
				}
				patch.Credential.Value = source.LLMGateway.value.Credential.value
				value["credential"] = patch.Credential.Value
			}
			operations++
		}
		if len(value) == 0 {
			return WorkerPatch{}, nil, 0, invalid("spec.worker.llmGateway cannot be empty")
		}
		result.LLMGateway = patch
		canonical["llmGateway"] = value
	}
	if source.Telemetry.present {
		patch, value, err := materializeTelemetry("spec.worker.telemetry", source.Telemetry)
		if err != nil {
			return WorkerPatch{}, nil, 0, err
		}
		result.Telemetry = patch
		canonical["telemetry"] = value
		operations++
	}
	if source.HTTPProxy.present {
		patch, value, err := materializeHTTPProxy(source.HTTPProxy)
		if err != nil {
			return WorkerPatch{}, nil, 0, err
		}
		result.HTTPProxy = patch
		canonical["httpProxy"] = value
		operations++
	}
	if source.Caido.present {
		patch, value, err := materializeCaido(source.Caido)
		if err != nil {
			return WorkerPatch{}, nil, 0, err
		}
		result.Caido = patch
		canonical["caido"] = value
		operations++
	}
	return result, canonical, operations, nil
}

func materializeTelemetry(path string, source optional[telemetrySource]) (AtomicPatch[TelemetryConfig], any, error) {
	patch := AtomicPatch[TelemetryConfig]{Present: true}
	if source.null {
		patch.Clear = true
		return patch, nil, nil
	}
	value := source.value
	if value.Adapter != "otlp-http@1" {
		return AtomicPatch[TelemetryConfig]{}, nil, invalid("%s.adapter must be otlp-http@1", path)
	}
	endpoint, err := validateURL(path+".endpoint", value.Endpoint)
	if err != nil {
		return AtomicPatch[TelemetryConfig]{}, nil, err
	}
	if value.Credential.present && value.Credential.null {
		return AtomicPatch[TelemetryConfig]{}, nil, invalid("%s.credential cannot be null", path)
	}
	if value.Credential.present {
		if err := validateID(path+".credential", value.Credential.value, 128); err != nil {
			return AtomicPatch[TelemetryConfig]{}, nil, err
		}
	}
	capture := false
	if value.CaptureContent.present {
		if value.CaptureContent.null || value.CaptureContent.value {
			return AtomicPatch[TelemetryConfig]{}, nil, invalid("%s.captureContent must be false", path)
		}
		capture = value.CaptureContent.value
	}
	flush := 3
	if value.FlushTimeoutSeconds.present {
		if value.FlushTimeoutSeconds.null || value.FlushTimeoutSeconds.value < 1 || value.FlushTimeoutSeconds.value > 10 {
			return AtomicPatch[TelemetryConfig]{}, nil, invalid("%s.flushTimeoutSeconds must be from 1 through 10", path)
		}
		flush = value.FlushTimeoutSeconds.value
	}
	patch.Value = TelemetryConfig{Adapter: value.Adapter, Endpoint: endpoint, Credential: value.Credential.value, CaptureContent: capture, FlushTimeoutSeconds: flush}
	canonical := map[string]any{"adapter": value.Adapter, "endpoint": endpoint, "captureContent": capture, "flushTimeoutSeconds": flush}
	if value.Credential.present {
		canonical["credential"] = value.Credential.value
	}
	return patch, canonical, nil
}

func materializeHTTPProxy(source optional[httpProxySource]) (AtomicPatch[HTTPProxyConfig], any, error) {
	patch := AtomicPatch[HTTPProxyConfig]{Present: true}
	if source.null {
		patch.Clear = true
		return patch, nil, nil
	}
	value := source.value
	if value.Adapter != "http-proxy@1" {
		return AtomicPatch[HTTPProxyConfig]{}, nil, invalid("spec.worker.httpProxy.adapter must be http-proxy@1")
	}
	proxyURL, err := validateURL("spec.worker.httpProxy.proxyUrl", value.ProxyURL)
	if err != nil {
		return AtomicPatch[HTTPProxyConfig]{}, nil, err
	}
	if value.Credential.present && value.Credential.null {
		return AtomicPatch[HTTPProxyConfig]{}, nil, invalid("spec.worker.httpProxy.credential cannot be null")
	}
	if value.Credential.present {
		if err := validateID("spec.worker.httpProxy.credential", value.Credential.value, 128); err != nil {
			return AtomicPatch[HTTPProxyConfig]{}, nil, err
		}
	}
	if value.CABundlePEM.present && value.CABundlePEM.null {
		return AtomicPatch[HTTPProxyConfig]{}, nil, invalid("spec.worker.httpProxy.caBundlePem cannot be null")
	}
	if value.CABundlePEM.present {
		if err := validateCABundle("spec.worker.httpProxy.caBundlePem", value.CABundlePEM.value); err != nil {
			return AtomicPatch[HTTPProxyConfig]{}, nil, err
		}
	}
	if len(value.Targets) == 0 {
		return AtomicPatch[HTTPProxyConfig]{}, nil, invalid("spec.worker.httpProxy.targets must not be empty")
	}
	allowed := map[string]bool{"llm-gateway": true, "tool-http": true, "tool-subprocess": true}
	seen := make(map[string]bool, len(value.Targets))
	for _, target := range value.Targets {
		if !allowed[target] || seen[target] {
			return AtomicPatch[HTTPProxyConfig]{}, nil, invalid("spec.worker.httpProxy.targets contains an invalid or duplicate target")
		}
		seen[target] = true
	}
	targets := append([]string(nil), value.Targets...)
	sort.Strings(targets)
	patch.Value = HTTPProxyConfig{Adapter: value.Adapter, ProxyURL: proxyURL, Credential: value.Credential.value, CABundlePEM: value.CABundlePEM.value, Targets: targets}
	canonical := map[string]any{"adapter": value.Adapter, "proxyUrl": proxyURL, "targets": targets}
	if value.Credential.present {
		canonical["credential"] = value.Credential.value
	}
	if value.CABundlePEM.present {
		canonical["caBundlePem"] = value.CABundlePEM.value
	}
	return patch, canonical, nil
}

func materializeCaido(source optional[caidoSource]) (AtomicPatch[CaidoConfig], any, error) {
	patch := AtomicPatch[CaidoConfig]{Present: true}
	if source.null {
		patch.Clear = true
		return patch, nil, nil
	}
	value := source.value
	if value.Adapter != string(contracts.RuntimeAdapterCaidoGraphQL) {
		return AtomicPatch[CaidoConfig]{}, nil, invalid("spec.worker.caido.adapter must be caido-graphql@1")
	}
	endpoint, err := validateURL("spec.worker.caido.endpoint", value.Endpoint)
	if err != nil {
		return AtomicPatch[CaidoConfig]{}, nil, err
	}
	if value.Credential.present && value.Credential.null {
		return AtomicPatch[CaidoConfig]{}, nil, invalid("spec.worker.caido.credential cannot be null")
	}
	if value.Credential.present {
		if err := validateID("spec.worker.caido.credential", value.Credential.value, 128); err != nil {
			return AtomicPatch[CaidoConfig]{}, nil, err
		}
	}
	if value.CABundlePEM.present && value.CABundlePEM.null {
		return AtomicPatch[CaidoConfig]{}, nil, invalid("spec.worker.caido.caBundlePem cannot be null")
	}
	if value.CABundlePEM.present {
		if err := validateCABundle("spec.worker.caido.caBundlePem", value.CABundlePEM.value); err != nil {
			return AtomicPatch[CaidoConfig]{}, nil, invalid("spec.worker.caido.caBundlePem is invalid")
		}
	}
	timeout := 0
	if value.RequestTimeoutSeconds.present {
		if value.RequestTimeoutSeconds.null || value.RequestTimeoutSeconds.value < 1 || value.RequestTimeoutSeconds.value > 120 {
			return AtomicPatch[CaidoConfig]{}, nil, invalid("spec.worker.caido.requestTimeoutSeconds must be from 1 through 120")
		}
		timeout = value.RequestTimeoutSeconds.value
	}
	patch.Value = CaidoConfig{
		Adapter: value.Adapter, Endpoint: endpoint, Credential: value.Credential.value,
		CABundlePEM: value.CABundlePEM.value, RequestTimeoutSeconds: timeout,
	}
	canonical := map[string]any{"adapter": value.Adapter, "endpoint": endpoint}
	if value.Credential.present {
		canonical["credential"] = value.Credential.value
	}
	if value.CABundlePEM.present {
		canonical["caBundlePem"] = value.CABundlePEM.value
	}
	if value.RequestTimeoutSeconds.present {
		canonical["requestTimeoutSeconds"] = timeout
	}
	return patch, canonical, nil
}

func collectGatewaySelectors(source specSource, result map[string]string) {
	if !source.Worker.present || source.Worker.null || !source.Worker.value.LLMGateway.present || source.Worker.value.LLMGateway.null {
		return
	}
	raw := source.Worker.value.LLMGateway.value.Gateway
	if !raw.present || raw.null {
		return
	}
	var selector string
	if json.Unmarshal(raw.value, &selector) == nil {
		result["spec.worker.llmGateway.gateway"] = selector
	}
}

func validateSelector(value string) error {
	if strings.Count(value, "@") != 1 {
		return ErrInvalid
	}
	id, version, _ := strings.Cut(value, "@")
	if !idPattern.MatchString(id) || len(id) > 63 || !versionPattern.MatchString(version) || len(version) > 128 {
		return ErrInvalid
	}
	return nil
}

func validateID(field, value string, maximum int) error {
	if !idPattern.MatchString(value) || len(value) > maximum {
		return invalid("%s is invalid", field)
	}
	return nil
}

func validateRef(ref Ref) error {
	if err := validateID("RuntimeConfig name", ref.Name, 63); err != nil || !versionPattern.MatchString(ref.Version) || len(ref.Version) > 128 || !digestPattern.MatchString(ref.Digest) {
		return invalid("RuntimeConfig ref is invalid")
	}
	return nil
}

func validateLabel(label string) error { return validateID("RuntimeConfig label", label, 63) }

func validateURL(field, raw string) (string, error) {
	if raw == "" || len([]byte(raw)) > maxURLBytes || raw != strings.TrimSpace(raw) || !utf8.ValidString(raw) {
		return "", invalid("%s is invalid", field)
	}
	parsed, err := url.Parse(raw)
	if err != nil || !parsed.IsAbs() || parsed.Opaque != "" || parsed.Host == "" || parsed.Hostname() == "" ||
		parsed.User != nil || parsed.Fragment != "" || parsed.RawFragment != "" || parsed.RawQuery != "" || parsed.ForceQuery ||
		(parsed.Scheme != "http" && parsed.Scheme != "https") {
		return "", invalid("%s must be an absolute HTTP(S) URL without userinfo, query, or fragment", field)
	}
	return parsed.String(), nil
}

func validateCABundle(field, raw string) error {
	if raw == "" || len([]byte(raw)) > maxPEMBytes || !utf8.ValidString(raw) {
		return invalid("%s is invalid", field)
	}
	rest := []byte(raw)
	count := 0
	for len(bytes.TrimSpace(rest)) > 0 {
		block, remaining := pem.Decode(rest)
		if block == nil {
			return invalid("%s contains non-PEM data", field)
		}
		if strings.Contains(block.Type, "PRIVATE KEY") || block.Type != "CERTIFICATE" {
			return invalid("%s must contain certificates only", field)
		}
		if _, err := x509.ParseCertificate(block.Bytes); err != nil {
			return invalid("%s contains an invalid certificate", field)
		}
		count++
		if count > 8 {
			return invalid("%s contains too many certificates", field)
		}
		rest = remaining
	}
	if count == 0 {
		return invalid("%s must contain a certificate", field)
	}
	return nil
}

func rejectDuplicateKeys(data []byte) error {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.UseNumber()
	if err := scanJSONValue(decoder, "$"); err != nil {
		return err
	}
	if token, err := decoder.Token(); !errors.Is(err, io.EOF) {
		if err == nil {
			_ = token
		}
		return invalid("document contains trailing JSON")
	}
	return nil
}

// encoding/json replaces isolated UTF-16 surrogate escapes with U+FFFD. JCS
// requires I-JSON scalar values instead, so reject them before decoding loses
// that distinction.
func rejectInvalidUnicodeEscapes(data []byte) error {
	insideString := false
	for index := 0; index < len(data); index++ {
		switch data[index] {
		case '"':
			insideString = !insideString
		case '\\':
			if !insideString || index+1 >= len(data) {
				continue
			}
			index++
			if data[index] != 'u' {
				continue
			}
			code, ok := parseHexQuad(data, index+1)
			if !ok {
				continue // The JSON decoder reports malformed escapes generically.
			}
			index += 4
			switch {
			case code >= 0xd800 && code <= 0xdbff:
				if index+6 >= len(data) || data[index+1] != '\\' || data[index+2] != 'u' {
					return invalid("document contains an invalid Unicode scalar value")
				}
				low, validLow := parseHexQuad(data, index+3)
				if !validLow || low < 0xdc00 || low > 0xdfff {
					return invalid("document contains an invalid Unicode scalar value")
				}
				index += 6
			case code >= 0xdc00 && code <= 0xdfff:
				return invalid("document contains an invalid Unicode scalar value")
			}
		}
	}
	return nil
}

func parseHexQuad(data []byte, start int) (uint16, bool) {
	if start+4 > len(data) {
		return 0, false
	}
	var result uint16
	for _, raw := range data[start : start+4] {
		result <<= 4
		switch {
		case raw >= '0' && raw <= '9':
			result += uint16(raw - '0')
		case raw >= 'a' && raw <= 'f':
			result += uint16(raw-'a') + 10
		case raw >= 'A' && raw <= 'F':
			result += uint16(raw-'A') + 10
		default:
			return 0, false
		}
	}
	return result, true
}

func scanJSONValue(decoder *json.Decoder, path string) error {
	token, err := decoder.Token()
	if err != nil {
		return invalid("document is not valid JSON")
	}
	delim, ok := token.(json.Delim)
	if !ok {
		return nil
	}
	switch delim {
	case '{':
		seen := make(map[string]struct{})
		for decoder.More() {
			keyToken, err := decoder.Token()
			if err != nil {
				return invalid("document is not valid JSON")
			}
			key, ok := keyToken.(string)
			if !ok {
				return invalid("document contains an invalid object key")
			}
			if _, exists := seen[key]; exists {
				return invalid("document contains a duplicate object key")
			}
			seen[key] = struct{}{}
			if err := scanJSONValue(decoder, path+"."+key); err != nil {
				return err
			}
		}
		if closeToken, err := decoder.Token(); err != nil || closeToken != json.Delim('}') {
			return invalid("document is not valid JSON")
		}
	case '[':
		index := 0
		for decoder.More() {
			if err := scanJSONValue(decoder, fmt.Sprintf("%s[%d]", path, index)); err != nil {
				return err
			}
			index++
		}
		if closeToken, err := decoder.Token(); err != nil || closeToken != json.Delim(']') {
			return invalid("document is not valid JSON")
		}
	default:
		return invalid("document is not valid JSON")
	}
	return nil
}

func decodeStrict(data []byte, target any) error {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(target); err != nil {
		return err
	}
	var trailing any
	if err := decoder.Decode(&trailing); !errors.Is(err, io.EOF) {
		return errors.New("multiple JSON values are not allowed")
	}
	return nil
}

func canonicalize(value any) ([]byte, error) {
	encoded, err := json.Marshal(value)
	if err != nil {
		return nil, err
	}
	var generic any
	if err := json.Unmarshal(encoded, &generic); err != nil {
		return nil, err
	}
	formatted, err := jcs.Format(generic)
	if err != nil {
		return nil, err
	}
	return []byte(formatted), nil
}

func digest(value []byte) string {
	sum := sha256.Sum256(value)
	return "sha256:" + hex.EncodeToString(sum[:])
}

func DigestIdempotencyKey(value string) (string, error) {
	if value == "" || len(value) > 128 || value != strings.TrimSpace(value) {
		return "", invalid("idempotency key is invalid")
	}
	return digest([]byte(value)), nil
}

func invalid(format string, args ...any) error {
	return fmt.Errorf("%w: %s", ErrInvalid, fmt.Sprintf(format, args...))
}
