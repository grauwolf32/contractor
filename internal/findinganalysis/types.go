// Package findinganalysis validates optional producer claims against the exact
// evidence of one receipt. It performs no network, model or filesystem lookup.
package findinganalysis

import "github.com/grauwolf32/contractor/internal/contracts"

const (
	MediaType        = "application/vnd.contractor.finding-analysis+json"
	Schema           = "contractor.finding.analysis.v1"
	maximumBytes     = 1 << 20
	maximumTextBytes = 16 << 10
)

type Document struct {
	Schema                string         `json:"schema" strictjson:"required"`
	ClientKey             string         `json:"client_key" strictjson:"required"`
	Sources               []Source       `json:"sources" strictjson:"required"`
	Locations             []Location     `json:"locations" strictjson:"required"`
	Flows                 []Flow         `json:"flows" strictjson:"required"`
	HTTPExchanges         []HTTPExchange `json:"http_exchanges" strictjson:"required"`
	Rule                  *Rule          `json:"rule,omitempty"`
	PrimaryLocationID     string         `json:"primary_location_id,omitempty"`
	PrimaryHTTPExchangeID string         `json:"primary_http_exchange_id,omitempty"`
}

type Rule struct {
	ID          string `json:"id" strictjson:"required"`
	Version     string `json:"version" strictjson:"required"`
	Name        string `json:"name" strictjson:"required"`
	Description string `json:"description" strictjson:"required"`
	HelpURI     string `json:"help_uri,omitempty"`
}

type Source struct {
	ID          string                `json:"id" strictjson:"required"`
	Kind        string                `json:"kind" strictjson:"required"`
	ArtifactRef contracts.ArtifactRef `json:"artifact_ref" strictjson:"required"`
}

type Location struct {
	ID         string  `json:"id" strictjson:"required"`
	SourceID   string  `json:"source_id" strictjson:"required"`
	Path       string  `json:"path" strictjson:"required"`
	FileDigest string  `json:"file_digest" strictjson:"required"`
	Region     *Region `json:"region,omitempty"`
	Symbol     *Symbol `json:"symbol,omitempty"`
}

type Region struct {
	StartLine   int64  `json:"start_line" strictjson:"required"`
	EndLine     *int64 `json:"end_line,omitempty"`
	StartColumn *int64 `json:"start_column,omitempty"`
	EndColumn   *int64 `json:"end_column,omitempty"`
}

type Symbol struct {
	Name          string `json:"name" strictjson:"required"`
	QualifiedName string `json:"qualified_name,omitempty"`
	Language      string `json:"language,omitempty"`
	Kind          string `json:"kind,omitempty"`
	Signature     string `json:"signature,omitempty"`
}

type Flow struct {
	ID           string                  `json:"id" strictjson:"required"`
	Kind         string                  `json:"kind" strictjson:"required"`
	Description  string                  `json:"description" strictjson:"required"`
	Completeness string                  `json:"completeness" strictjson:"required"`
	Steps        []Step                  `json:"steps" strictjson:"required"`
	Edges        []Edge                  `json:"edges" strictjson:"required"`
	Gaps         []Gap                   `json:"gaps" strictjson:"required"`
	EvidenceRefs []contracts.ArtifactRef `json:"evidence_refs" strictjson:"required"`
}

type Step struct {
	ID           string                  `json:"id" strictjson:"required"`
	Roles        []string                `json:"roles" strictjson:"required"`
	Message      string                  `json:"message" strictjson:"required"`
	EvidenceRefs []contracts.ArtifactRef `json:"evidence_refs" strictjson:"required"`
	LocationID   string                  `json:"location_id,omitempty"`
	Value        string                  `json:"value,omitempty"`
	ValueState   string                  `json:"value_state,omitempty"`
}

type Edge struct {
	From            string                  `json:"from" strictjson:"required"`
	To              string                  `json:"to" strictjson:"required"`
	Relation        string                  `json:"relation" strictjson:"required"`
	Basis           string                  `json:"basis" strictjson:"required"`
	EvidenceRefs    []contracts.ArtifactRef `json:"evidence_refs" strictjson:"required"`
	ArgumentBinding *ArgumentBinding        `json:"argument_binding,omitempty"`
}

type ArgumentBinding struct {
	CallerValue     string `json:"caller_value" strictjson:"required"`
	CalleeParameter string `json:"callee_parameter" strictjson:"required"`
}

type Gap struct {
	AfterStepID  string `json:"after_step_id" strictjson:"required"`
	BeforeStepID string `json:"before_step_id" strictjson:"required"`
	Reason       string `json:"reason" strictjson:"required"`
}

type HTTPExchange struct {
	ID                       string                `json:"id" strictjson:"required"`
	EvidenceRef              contracts.ArtifactRef `json:"evidence_ref" strictjson:"required"`
	Decoder                  string                `json:"decoder" strictjson:"required"`
	Operation                *Operation            `json:"operation,omitempty"`
	ImplementationLocationID string                `json:"implementation_location_id,omitempty"`
}

type Operation struct {
	DocumentRef  contracts.ArtifactRef `json:"document_ref" strictjson:"required"`
	Pointer      string                `json:"pointer" strictjson:"required"`
	Method       string                `json:"method" strictjson:"required"`
	PathTemplate string                `json:"path_template" strictjson:"required"`
}
