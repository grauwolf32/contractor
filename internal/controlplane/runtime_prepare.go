package controlplane

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/url"
	"slices"
	"strings"
	"time"

	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/control"
	"github.com/grauwolf32/contractor/internal/contracts/reporting"
	"github.com/grauwolf32/contractor/internal/contracts/runtimesettings"
	"github.com/grauwolf32/contractor/internal/helpers/secretmatch"
)

func (c *RuntimeControlClient) Prepare(
	ctx context.Context,
	reservation Reservation,
	settings runtimesettings.WorkerExecutionSettings,
) (contracts.WorkerHandle, error) {
	if !contracts.SupportsWorkerCompletion(reservation.CompletionCapabilities, reservation.CompletionContract) {
		return contracts.WorkerHandle{}, fmt.Errorf("%w: Runtime does not support completion contract", ErrInvalidRequest)
	}
	if err := resolved.ValidateWorkerCompletionSelection(reservation.CompletionContract, reservation.Grant.Namespace, reservation.AgentTemplate, reservation.ContextManagement); err != nil {
		return contracts.WorkerHandle{}, err
	}
	resolvedSkills := reservation.ResolvedSkills
	if resolvedSkills == nil && len(reservation.AgentTemplate.Skills) == 0 {
		resolvedSkills = []contracts.ResolvedSkill{}
	}
	spec := control.AllocationSpec{
		CompletionContract: contracts.CloneWorkerCompletionContract(reservation.CompletionContract),
		APIVersion:         contracts.APIVersion, AllocationID: reservation.Grant.AllocationID,
		RunID: reservation.Grant.RunID, StageExecutionID: reservation.Grant.StageExecutionID,
		LogicalAgentName: reservation.Grant.LogicalAgentName, Namespace: reservation.Grant.Namespace,
		WorkerSessionMode: reservation.WorkerSessionMode,
		ContextManagement: reservation.ContextManagement.Clone(),
		RunMetadataLabels: reservation.RunMetadataLabels.Clone(),
		LeaseExpiresAt:    wireTime(reservation.LeaseExpiresAt), AgentTemplate: reservation.AgentTemplate.Clone(),
		ResolvedSkills: contracts.CloneResolvedSkills(resolvedSkills),
		ModelPolicy:    settings.ModelPolicy.Clone(), RuntimeSettings: settings.RuntimeSettings,
		SummarizerModelPolicy:           settings.SummarizerModelPolicy,
		ResolvedRuntimeConfigProvenance: settings.ResolvedRuntimeConfigProvenance,
		Workspace:                       contracts.CloneAllocationWorkspaceSpec(reservation.Workspace),
		PerformanceMetrics:              clonePerformanceMetricsRequest(reservation.PerformanceMetrics),
	}
	if c.preparation != nil {
		return c.preparation.WithPreparedSpec(ctx, reservation, spec, func(prepared control.AllocationSpec) (contracts.WorkerHandle, error) {
			return c.prepareAllocation(ctx, reservation, prepared)
		})
	}
	return c.prepareAllocation(ctx, reservation, spec)
}

func (c *RuntimeControlClient) prepareAllocation(ctx context.Context, reservation Reservation, spec control.AllocationSpec) (contracts.WorkerHandle, error) {
	request := control.PrepareAllocationRequest{APIVersion: contracts.APIVersion, Spec: spec}
	if err := request.Validate(); err != nil {
		return contracts.WorkerHandle{}, fmt.Errorf("build prepare request: %w", err)
	}
	if c.grantReader != nil {
		current, err := c.grantReader.GetGrant(reservation.Grant.AllocationID)
		if err != nil {
			return contracts.WorkerHandle{}, err
		}
		expected := reservation.Grant
		if current.WriteFenced || current.Lost || current.AllocationID != expected.AllocationID || current.RunID != expected.RunID || current.StageExecutionID != expected.StageExecutionID || current.RuntimeAgentID != expected.RuntimeAgentID || current.RuntimeInstanceID != expected.RuntimeInstanceID || current.Namespace != expected.Namespace || current.LogicalAgentName != expected.LogicalAgentName {
			return contracts.WorkerHandle{}, fmt.Errorf("%w: allocation authority changed during preparation", ErrInvalidRequest)
		}
	} else if c.preparation != nil {
		return contracts.WorkerHandle{}, fmt.Errorf("%w: allocation grant reader is unavailable", ErrInvalidRequest)
	}
	if err := ctx.Err(); err != nil {
		return contracts.WorkerHandle{}, err
	}
	var response control.PrepareAllocationResponse
	if err := c.postJSON(
		ctx, reservation.ControlURL, reservation.Grant.AllocationID,
		reservation.Grant.RuntimeAgentID, "prepare", request, &response, true,
	); err != nil {
		return contracts.WorkerHandle{}, err
	}
	if err := validateWorkerHandle(response.WorkerHandle, reservation, spec.RuntimeSettings); err != nil {
		return contracts.WorkerHandle{}, err
	}
	response.WorkerHandle.RuntimeAgentID = reservation.Grant.RuntimeAgentID
	return response.WorkerHandle, nil
}

func clonePerformanceMetricsRequest(source *reporting.PerformanceMetricsRequest) *reporting.PerformanceMetricsRequest {
	if source == nil {
		return nil
	}
	result := *source
	return &result
}

func validateWorkerHandle(
	handle contracts.WorkerHandle,
	reservation Reservation,
	settings runtimesettings.RuntimeSettings,
) error {
	minimumLease := reservation.initialLeaseExpiresAt
	if minimumLease.IsZero() {
		minimumLease = reservation.LeaseExpiresAt
	}
	if handle.AllocationID != reservation.Grant.AllocationID ||
		handle.AgentTemplateRef != reservation.AgentTemplate.Ref ||
		handle.WorkerRuntimeRef != reservation.AgentTemplate.Runtime ||
		handle.LeaseExpiresAt.Before(wireTime(minimumLease)) ||
		handle.LeaseExpiresAt.After(wireTime(reservation.LeaseExpiresAt)) {
		return errors.New("Runtime Agent returned a WorkerHandle for different resolved inputs")
	}
	if _, err := json.Marshal(handle); err != nil {
		return errors.New("Runtime Agent returned an unencodable WorkerHandle")
	}
	cardValues, cardKeys := workerHandleUntrustedCardText(handle.AgentCard, reservation)
	secrets := settings.SecretValues()
	for value := range cardValues {
		if secretmatch.Contains(value, secrets) {
			return errors.New("Runtime Agent exposed RuntimeSettings secret in WorkerHandle")
		}
	}
	// Keys are protocol vocabulary, so only a long secret is looked for in them.
	long := secretmatch.Long(secrets)
	for key := range cardKeys {
		for _, secret := range long {
			if strings.Contains(key, secret) {
				return errors.New("Runtime Agent exposed RuntimeSettings secret in WorkerHandle")
			}
		}
	}
	if err := validateA2AAgentCard(
		handle.AgentCard, reservation.Grant.AllocationID, reservation.A2AURL,
	); err != nil {
		return err
	}
	return nil
}

// workerHandleUntrustedCardText returns the Agent Card values to scan and every
// object key. The handle identity is validated against the reservation, and
// workerCardTrustedValues lists the exact values fixed by the Server/Runtime
// protocol at their paths; a credential coinciding with one of them is not a
// leak. Other values remain checked, including changed values at a normally
// fixed path. Keys are scanned whatever their path, but only for secrets long
// enough to match anywhere, because nearly all of them are protocol vocabulary.
func workerHandleUntrustedCardText(
	card map[string]any,
	reservation Reservation,
) (values, keys map[string]struct{}) {
	values = make(map[string]struct{})
	keys = make(map[string]struct{})
	collectUntrustedCardText(values, keys, card, nil, workerCardTrustedValues(reservation))
	return values, keys
}

func collectUntrustedCardText(
	values, keys map[string]struct{},
	value any,
	path []string,
	trusted map[string][]string,
) {
	switch typed := value.(type) {
	case string:
		encodedPath, _ := json.Marshal(path)
		if !slices.Contains(trusted[string(encodedPath)], typed) {
			values[typed] = struct{}{}
		}
	case []any:
		for index, item := range typed {
			collectUntrustedCardText(values, keys, item, append(path, fmt.Sprint(index)), trusted)
		}
	case []string:
		for index, item := range typed {
			collectUntrustedCardText(values, keys, item, append(path, fmt.Sprint(index)), trusted)
		}
	case map[string]any:
		for key, item := range typed {
			keys[key] = struct{}{}
			collectUntrustedCardText(values, keys, item, append(path, key), trusted)
		}
	}
}

// workerCardTrustedValues lists, by JSON-encoded card path, the exact Agent Card
// strings fixed by the Server/Runtime protocol or taken from the reservation.
// api/testdata/v1alpha1/agent-card-secret-scan-cases.json holds the same table,
// which the Python Runtime's Worker handle check shares.
func workerCardTrustedValues(reservation Reservation) map[string][]string {
	grant := reservation.Grant
	endpoint := strings.TrimRight(reservation.A2AURL, "/") +
		"/private/v1/allocations/" + grant.AllocationID + "/a2a"
	return map[string][]string{
		`["name"]`:                          {grant.LogicalAgentName, "Contractor Worker " + grant.LogicalAgentName},
		`["description"]`:                   {reservation.AgentTemplate.Description},
		`["version"]`:                       {reservation.AgentTemplate.Ref.Version},
		`["url"]`:                           {endpoint},
		`["protocolVersion"]`:               {"1.0"},
		`["supportedInterfaces","0","url"]`: {endpoint},
		`["supportedInterfaces","0","protocolBinding"]`: {"JSONRPC"},
		`["supportedInterfaces","0","protocolVersion"]`: {"1.0"},
		`["supportedInterfaces","0","tenant"]`:          {grant.AllocationID},
		`["defaultInputModes","0"]`:                     {stageContentMediaType, "application/json"},
		`["defaultOutputModes","0"]`:                    {workerCompletionMediaType, "application/json"},
		`["skills","0","id"]`:                           {"contractor_stage_content"},
		`["skills","0","name"]`:                         {"Execute Contractor stage content"},
		`["skills","0","description"]`:                  {"Execute one strict Contractor StageContentRequest."},
		`["skills","0","tags","0"]`:                     {"contractor"},
		`["skills","0","tags","1"]`:                     {"stage"},
		`["skills","0","inputModes","0"]`:               {stageContentMediaType},
		`["skills","0","outputModes","0"]`:              {workerCompletionMediaType},
		`["securitySchemes","mutualTLS","mtlsSecurityScheme","description"]`: {
			"Deployment-CA mutual TLS with a Contractor Control Plane peer",
		},
	}
}

const (
	stageContentMediaType     = "application/vnd.contractor.stage-content+json"
	workerCompletionMediaType = "application/vnd.contractor.worker-completion+json"
)

func validateA2AAgentCard(card map[string]any, allocationID, registeredURL string) error {
	interfaces, ok := card["supportedInterfaces"].([]any)
	if !ok || len(interfaces) != 1 {
		return errors.New("Runtime Agent returned an incompatible A2A Agent Card")
	}
	current, ok := interfaces[0].(map[string]any)
	if !ok || current["protocolBinding"] != "JSONRPC" || current["protocolVersion"] != "1.0" ||
		current["tenant"] != allocationID {
		return errors.New("Runtime Agent returned an incompatible A2A Agent Card")
	}
	cardURL, ok := current["url"].(string)
	if !ok || !isExpectedA2AEndpoint(cardURL, registeredURL, allocationID) {
		return errors.New("Runtime Agent returned an A2A Agent Card for another endpoint")
	}
	if !oneStringValue(card["defaultInputModes"], stageContentMediaType) ||
		!oneStringValue(card["defaultOutputModes"], workerCompletionMediaType) {
		return errors.New("Runtime Agent returned incompatible A2A content modes")
	}
	skills, ok := card["skills"].([]any)
	if !ok || len(skills) != 1 {
		return errors.New("Runtime Agent returned an incompatible A2A skill set")
	}
	skill, ok := skills[0].(map[string]any)
	if !ok || skill["id"] != "contractor_stage_content" ||
		!oneStringValue(skill["inputModes"], stageContentMediaType) ||
		!oneStringValue(skill["outputModes"], workerCompletionMediaType) {
		return errors.New("Runtime Agent returned an incompatible A2A skill set")
	}
	securitySchemes, ok := card["securitySchemes"].(map[string]any)
	if !ok {
		return errors.New("Runtime Agent A2A Agent Card does not declare mutual TLS")
	}
	mutualTLS, ok := securitySchemes["mutualTLS"].(map[string]any)
	if !ok {
		return errors.New("Runtime Agent A2A Agent Card does not declare mutual TLS")
	}
	if _, ok := mutualTLS["mtlsSecurityScheme"].(map[string]any); !ok {
		return errors.New("Runtime Agent A2A Agent Card does not declare mutual TLS")
	}
	securityRequirements, ok := card["securityRequirements"].([]any)
	if !ok || len(securityRequirements) != 1 {
		return errors.New("Runtime Agent A2A Agent Card does not require mutual TLS")
	}
	requirement, ok := securityRequirements[0].(map[string]any)
	if !ok {
		return errors.New("Runtime Agent A2A Agent Card does not require mutual TLS")
	}
	requiredSchemes, ok := requirement["schemes"].(map[string]any)
	if !ok || len(requiredSchemes) != 1 {
		return errors.New("Runtime Agent A2A Agent Card does not require mutual TLS")
	}
	if _, ok := requiredSchemes["mutualTLS"].(map[string]any); !ok {
		return errors.New("Runtime Agent A2A Agent Card does not require mutual TLS")
	}
	return nil
}

func oneStringValue(value any, expected string) bool {
	values, ok := value.([]any)
	return ok && len(values) == 1 && values[0] == expected
}

func isExpectedA2AEndpoint(cardURL, registeredURL, allocationID string) bool {
	card, cardErr := url.Parse(cardURL)
	registered, registeredErr := url.Parse(registeredURL)
	if cardErr != nil || registeredErr != nil || !card.IsAbs() || !registered.IsAbs() ||
		card.User != nil || card.RawQuery != "" || card.Fragment != "" ||
		registered.User != nil || registered.RawQuery != "" || registered.Fragment != "" ||
		!runtimePathIDPattern.MatchString(allocationID) {
		return false
	}
	expected := strings.TrimRight(registeredURL, "/") +
		"/private/v1/allocations/" + allocationID + "/a2a"
	return cardURL == expected
}

// Python datetime and PostgreSQL both retain microseconds. Normalize private
// wire timestamps at the Go boundary so an echoed deadline remains exact.
func wireTime(value time.Time) time.Time {
	return value.UTC().Truncate(time.Microsecond)
}
