package public

// Projects: the management port and the Project request/response bodies.
// The custom UnmarshalJSON methods reject unknown and conflicting fields
// before any value reaches the service.

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/projectstore"
)

type ProjectManagement interface {
	Create(context.Context, projectstore.CreateParams) (projectstore.Project, bool, error)
	Get(context.Context, string, string) (projectstore.Project, error)
	List(context.Context, projectstore.ListParams) ([]projectstore.Project, error)
	Update(context.Context, projectstore.UpdateParams) (projectstore.Project, error)
	BeginDeletion(context.Context, projectstore.BeginDeletionParams) (projectstore.Project, bool, error)
}

type createProjectRequest struct {
	Kind        projectstore.Kind `json:"kind"`
	Name        string            `json:"name"`
	Description string            `json:"description,omitempty"`
}

func (r *createProjectRequest) UnmarshalJSON(data []byte) error {
	type wire createProjectRequest
	var decoded wire
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&decoded); err != nil {
		return err
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(data, &fields); err != nil {
		return err
	}
	for _, required := range []string{"kind", "name"} {
		value, present := fields[required]
		if !present || bytes.Equal(bytes.TrimSpace(value), []byte("null")) {
			return fmt.Errorf("%s is required", required)
		}
	}
	if value, present := fields["description"]; present && bytes.Equal(bytes.TrimSpace(value), []byte("null")) {
		return errors.New("description cannot be null")
	}
	*r = createProjectRequest(decoded)
	return nil
}

type updateProjectRequest struct {
	Name        *string                   `json:"name,omitempty"`
	Description *string                   `json:"description,omitempty"`
	HTTPTarget  *projectHTTPTargetRequest `json:"httpTarget,omitempty"`
	targetSet   bool
}

type projectHTTPTargetRequest struct {
	URL        string                          `json:"url"`
	Credential *contracts.RuntimeCredentialRef `json:"credential,omitempty"`
}

func (r *updateProjectRequest) UnmarshalJSON(data []byte) error {
	type wire updateProjectRequest
	var decoded wire
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&decoded); err != nil {
		return err
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(data, &fields); err != nil {
		return err
	}
	if len(fields) == 0 {
		return errors.New("at least one Project field is required")
	}
	for name, value := range fields {
		if name != "httpTarget" && bytes.Equal(bytes.TrimSpace(value), []byte("null")) {
			return fmt.Errorf("%s cannot be null", name)
		}
	}
	if raw, present := fields["httpTarget"]; present {
		decoded.targetSet = true
		if !bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
			var target projectHTTPTargetRequest
			if err := decodeStrictPublicJSON(raw, &target); err != nil {
				return err
			}
			var targetFields map[string]json.RawMessage
			if err := json.Unmarshal(raw, &targetFields); err != nil {
				return err
			}
			urlValue, hasURL := targetFields["url"]
			if !hasURL || bytes.Equal(bytes.TrimSpace(urlValue), []byte("null")) {
				return errors.New("httpTarget.url is required")
			}
			candidate := contracts.HTTPOriginTargetRef{URL: target.URL, Credential: target.Credential}
			if err := candidate.Validate(); err != nil {
				return err
			}
			decoded.HTTPTarget = &target
		}
	}
	*r = updateProjectRequest(decoded)
	return nil
}

type projectResponse struct {
	ProjectID   string                         `json:"projectId"`
	Kind        projectstore.Kind              `json:"kind"`
	Name        string                         `json:"name"`
	Description string                         `json:"description"`
	HTTPTarget  *contracts.HTTPOriginTargetRef `json:"httpTarget,omitempty"`
	Lifecycle   projectstore.Lifecycle         `json:"lifecycle"`
	Deletion    *projectDeletionResponse       `json:"deletion,omitempty"`
	Revision    string                         `json:"revision"`
	CreatedAt   time.Time                      `json:"createdAt"`
	UpdatedAt   time.Time                      `json:"updatedAt"`
}

type projectDeletionResponse struct {
	Phase       projectstore.DeletionPhase `json:"phase"`
	RequestedAt time.Time                  `json:"requestedAt"`
}

type projectPageResponse struct {
	Items []projectResponse `json:"items"`
	Page  pageInfoResponse  `json:"page"`
}
