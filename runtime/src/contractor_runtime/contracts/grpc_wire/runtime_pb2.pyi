from google.protobuf import struct_pb2 as _struct_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class BootstrapRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class SharedConfig(_message.Message):
    __slots__ = ("api_version", "version", "limits", "digest")
    API_VERSION_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    LIMITS_FIELD_NUMBER: _ClassVar[int]
    DIGEST_FIELD_NUMBER: _ClassVar[int]
    api_version: str
    version: int
    limits: SharedLimits
    digest: str
    def __init__(self, api_version: _Optional[str] = ..., version: _Optional[int] = ..., limits: _Optional[_Union[SharedLimits, _Mapping]] = ..., digest: _Optional[str] = ...) -> None: ...

class SharedLimits(_message.Message):
    __slots__ = ("git_import_url_max_bytes", "run_metadata_max_labels", "run_metadata_key_max_bytes", "run_metadata_value_max_bytes", "audit_max_labels", "http_request_body_max_bytes", "http_response_body_max_bytes", "http_header_block_max_bytes", "worker_result_max_bytes", "worker_completion_max_bytes", "worker_failure_message_max_bytes", "worker_result_max_artifacts", "worker_observation_max_tools", "worker_workspace_max_files_read", "worker_state_max_bytes", "worker_state_workspace_max_paths", "worker_state_workspace_paths_max_bytes", "artifact_payload_max_bytes", "public_page_max_items", "artifact_list_max_items")
    GIT_IMPORT_URL_MAX_BYTES_FIELD_NUMBER: _ClassVar[int]
    RUN_METADATA_MAX_LABELS_FIELD_NUMBER: _ClassVar[int]
    RUN_METADATA_KEY_MAX_BYTES_FIELD_NUMBER: _ClassVar[int]
    RUN_METADATA_VALUE_MAX_BYTES_FIELD_NUMBER: _ClassVar[int]
    AUDIT_MAX_LABELS_FIELD_NUMBER: _ClassVar[int]
    HTTP_REQUEST_BODY_MAX_BYTES_FIELD_NUMBER: _ClassVar[int]
    HTTP_RESPONSE_BODY_MAX_BYTES_FIELD_NUMBER: _ClassVar[int]
    HTTP_HEADER_BLOCK_MAX_BYTES_FIELD_NUMBER: _ClassVar[int]
    WORKER_RESULT_MAX_BYTES_FIELD_NUMBER: _ClassVar[int]
    WORKER_COMPLETION_MAX_BYTES_FIELD_NUMBER: _ClassVar[int]
    WORKER_FAILURE_MESSAGE_MAX_BYTES_FIELD_NUMBER: _ClassVar[int]
    WORKER_RESULT_MAX_ARTIFACTS_FIELD_NUMBER: _ClassVar[int]
    WORKER_OBSERVATION_MAX_TOOLS_FIELD_NUMBER: _ClassVar[int]
    WORKER_WORKSPACE_MAX_FILES_READ_FIELD_NUMBER: _ClassVar[int]
    WORKER_STATE_MAX_BYTES_FIELD_NUMBER: _ClassVar[int]
    WORKER_STATE_WORKSPACE_MAX_PATHS_FIELD_NUMBER: _ClassVar[int]
    WORKER_STATE_WORKSPACE_PATHS_MAX_BYTES_FIELD_NUMBER: _ClassVar[int]
    ARTIFACT_PAYLOAD_MAX_BYTES_FIELD_NUMBER: _ClassVar[int]
    PUBLIC_PAGE_MAX_ITEMS_FIELD_NUMBER: _ClassVar[int]
    ARTIFACT_LIST_MAX_ITEMS_FIELD_NUMBER: _ClassVar[int]
    git_import_url_max_bytes: int
    run_metadata_max_labels: int
    run_metadata_key_max_bytes: int
    run_metadata_value_max_bytes: int
    audit_max_labels: int
    http_request_body_max_bytes: int
    http_response_body_max_bytes: int
    http_header_block_max_bytes: int
    worker_result_max_bytes: int
    worker_completion_max_bytes: int
    worker_failure_message_max_bytes: int
    worker_result_max_artifacts: int
    worker_observation_max_tools: int
    worker_workspace_max_files_read: int
    worker_state_max_bytes: int
    worker_state_workspace_max_paths: int
    worker_state_workspace_paths_max_bytes: int
    artifact_payload_max_bytes: int
    public_page_max_items: int
    artifact_list_max_items: int
    def __init__(self, git_import_url_max_bytes: _Optional[int] = ..., run_metadata_max_labels: _Optional[int] = ..., run_metadata_key_max_bytes: _Optional[int] = ..., run_metadata_value_max_bytes: _Optional[int] = ..., audit_max_labels: _Optional[int] = ..., http_request_body_max_bytes: _Optional[int] = ..., http_response_body_max_bytes: _Optional[int] = ..., http_header_block_max_bytes: _Optional[int] = ..., worker_result_max_bytes: _Optional[int] = ..., worker_completion_max_bytes: _Optional[int] = ..., worker_failure_message_max_bytes: _Optional[int] = ..., worker_result_max_artifacts: _Optional[int] = ..., worker_observation_max_tools: _Optional[int] = ..., worker_workspace_max_files_read: _Optional[int] = ..., worker_state_max_bytes: _Optional[int] = ..., worker_state_workspace_max_paths: _Optional[int] = ..., worker_state_workspace_paths_max_bytes: _Optional[int] = ..., artifact_payload_max_bytes: _Optional[int] = ..., public_page_max_items: _Optional[int] = ..., artifact_list_max_items: _Optional[int] = ...) -> None: ...

class AgentHeartbeat(_message.Message):
    __slots__ = ("api_version", "instance_id", "heartbeat_seq", "echoed_ack_seq", "observed_state", "allocations")
    API_VERSION_FIELD_NUMBER: _ClassVar[int]
    INSTANCE_ID_FIELD_NUMBER: _ClassVar[int]
    HEARTBEAT_SEQ_FIELD_NUMBER: _ClassVar[int]
    ECHOED_ACK_SEQ_FIELD_NUMBER: _ClassVar[int]
    OBSERVED_STATE_FIELD_NUMBER: _ClassVar[int]
    ALLOCATIONS_FIELD_NUMBER: _ClassVar[int]
    api_version: str
    instance_id: str
    heartbeat_seq: int
    echoed_ack_seq: int
    observed_state: str
    allocations: ListOfObservedAllocation
    def __init__(self, api_version: _Optional[str] = ..., instance_id: _Optional[str] = ..., heartbeat_seq: _Optional[int] = ..., echoed_ack_seq: _Optional[int] = ..., observed_state: _Optional[str] = ..., allocations: _Optional[_Union[ListOfObservedAllocation, _Mapping]] = ...) -> None: ...

class HeartbeatResponse(_message.Message):
    __slots__ = ("api_version", "ack_seq", "action", "allocation_actions")
    API_VERSION_FIELD_NUMBER: _ClassVar[int]
    ACK_SEQ_FIELD_NUMBER: _ClassVar[int]
    ACTION_FIELD_NUMBER: _ClassVar[int]
    ALLOCATION_ACTIONS_FIELD_NUMBER: _ClassVar[int]
    api_version: str
    ack_seq: int
    action: str
    allocation_actions: ListOfAllocationAction
    def __init__(self, api_version: _Optional[str] = ..., ack_seq: _Optional[int] = ..., action: _Optional[str] = ..., allocation_actions: _Optional[_Union[ListOfAllocationAction, _Mapping]] = ...) -> None: ...

class AllocationAction(_message.Message):
    __slots__ = ("allocation_id", "action")
    ALLOCATION_ID_FIELD_NUMBER: _ClassVar[int]
    ACTION_FIELD_NUMBER: _ClassVar[int]
    allocation_id: str
    action: str
    def __init__(self, allocation_id: _Optional[str] = ..., action: _Optional[str] = ...) -> None: ...

class ListOfAllocationAction(_message.Message):
    __slots__ = ("values",)
    VALUES_FIELD_NUMBER: _ClassVar[int]
    values: _containers.RepeatedCompositeFieldContainer[AllocationAction]
    def __init__(self, values: _Optional[_Iterable[_Union[AllocationAction, _Mapping]]] = ...) -> None: ...

class ListOfString(_message.Message):
    __slots__ = ("values",)
    VALUES_FIELD_NUMBER: _ClassVar[int]
    values: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, values: _Optional[_Iterable[str]] = ...) -> None: ...

class ListOfInteger(_message.Message):
    __slots__ = ("values",)
    VALUES_FIELD_NUMBER: _ClassVar[int]
    values: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, values: _Optional[_Iterable[int]] = ...) -> None: ...

class ListOfToolsetCapability(_message.Message):
    __slots__ = ("values",)
    VALUES_FIELD_NUMBER: _ClassVar[int]
    values: _containers.RepeatedCompositeFieldContainer[ToolsetCapability]
    def __init__(self, values: _Optional[_Iterable[_Union[ToolsetCapability, _Mapping]]] = ...) -> None: ...

class ListOfObservedAllocation(_message.Message):
    __slots__ = ("values",)
    VALUES_FIELD_NUMBER: _ClassVar[int]
    values: _containers.RepeatedCompositeFieldContainer[ObservedAllocation]
    def __init__(self, values: _Optional[_Iterable[_Union[ObservedAllocation, _Mapping]]] = ...) -> None: ...

class AgentRegistration(_message.Message):
    __slots__ = ("api_version", "instance_id", "software_version", "started_at", "control_url", "a2a_url", "supported_runtimes", "supported_toolsets", "supported_sandbox_profiles", "observed_state", "capabilities", "supported_runtime_adapters", "workspace_capabilities", "supported_performance_metrics_versions", "placement_labels", "allocation_slots", "allocations")
    API_VERSION_FIELD_NUMBER: _ClassVar[int]
    INSTANCE_ID_FIELD_NUMBER: _ClassVar[int]
    SOFTWARE_VERSION_FIELD_NUMBER: _ClassVar[int]
    STARTED_AT_FIELD_NUMBER: _ClassVar[int]
    CONTROL_URL_FIELD_NUMBER: _ClassVar[int]
    A2A_URL_FIELD_NUMBER: _ClassVar[int]
    SUPPORTED_RUNTIMES_FIELD_NUMBER: _ClassVar[int]
    SUPPORTED_TOOLSETS_FIELD_NUMBER: _ClassVar[int]
    SUPPORTED_SANDBOX_PROFILES_FIELD_NUMBER: _ClassVar[int]
    OBSERVED_STATE_FIELD_NUMBER: _ClassVar[int]
    CAPABILITIES_FIELD_NUMBER: _ClassVar[int]
    SUPPORTED_RUNTIME_ADAPTERS_FIELD_NUMBER: _ClassVar[int]
    WORKSPACE_CAPABILITIES_FIELD_NUMBER: _ClassVar[int]
    SUPPORTED_PERFORMANCE_METRICS_VERSIONS_FIELD_NUMBER: _ClassVar[int]
    PLACEMENT_LABELS_FIELD_NUMBER: _ClassVar[int]
    ALLOCATION_SLOTS_FIELD_NUMBER: _ClassVar[int]
    ALLOCATIONS_FIELD_NUMBER: _ClassVar[int]
    api_version: str
    instance_id: str
    software_version: str
    started_at: str
    control_url: str
    a2a_url: str
    supported_runtimes: ListOfString
    supported_toolsets: ListOfToolsetCapability
    supported_sandbox_profiles: ListOfString
    observed_state: str
    capabilities: NullableCompletionCapabilities
    supported_runtime_adapters: ListOfString
    workspace_capabilities: NullableWorkspaceCapabilities
    supported_performance_metrics_versions: ListOfInteger
    placement_labels: ListOfString
    allocation_slots: int
    allocations: ListOfObservedAllocation
    def __init__(self, api_version: _Optional[str] = ..., instance_id: _Optional[str] = ..., software_version: _Optional[str] = ..., started_at: _Optional[str] = ..., control_url: _Optional[str] = ..., a2a_url: _Optional[str] = ..., supported_runtimes: _Optional[_Union[ListOfString, _Mapping]] = ..., supported_toolsets: _Optional[_Union[ListOfToolsetCapability, _Mapping]] = ..., supported_sandbox_profiles: _Optional[_Union[ListOfString, _Mapping]] = ..., observed_state: _Optional[str] = ..., capabilities: _Optional[_Union[NullableCompletionCapabilities, _Mapping]] = ..., supported_runtime_adapters: _Optional[_Union[ListOfString, _Mapping]] = ..., workspace_capabilities: _Optional[_Union[NullableWorkspaceCapabilities, _Mapping]] = ..., supported_performance_metrics_versions: _Optional[_Union[ListOfInteger, _Mapping]] = ..., placement_labels: _Optional[_Union[ListOfString, _Mapping]] = ..., allocation_slots: _Optional[int] = ..., allocations: _Optional[_Union[ListOfObservedAllocation, _Mapping]] = ...) -> None: ...

class AgentRegistrationResponse(_message.Message):
    __slots__ = ("api_version", "heartbeat_interval_seconds", "confirmed_lease_seconds")
    API_VERSION_FIELD_NUMBER: _ClassVar[int]
    HEARTBEAT_INTERVAL_SECONDS_FIELD_NUMBER: _ClassVar[int]
    CONFIRMED_LEASE_SECONDS_FIELD_NUMBER: _ClassVar[int]
    api_version: str
    heartbeat_interval_seconds: int
    confirmed_lease_seconds: int
    def __init__(self, api_version: _Optional[str] = ..., heartbeat_interval_seconds: _Optional[int] = ..., confirmed_lease_seconds: _Optional[int] = ...) -> None: ...

class ToolsetCapability(_message.Message):
    __slots__ = ("ref", "tools")
    REF_FIELD_NUMBER: _ClassVar[int]
    TOOLS_FIELD_NUMBER: _ClassVar[int]
    ref: str
    tools: ListOfString
    def __init__(self, ref: _Optional[str] = ..., tools: _Optional[_Union[ListOfString, _Mapping]] = ...) -> None: ...

class ObservedAllocation(_message.Message):
    __slots__ = ("allocation_id", "observed_state")
    ALLOCATION_ID_FIELD_NUMBER: _ClassVar[int]
    OBSERVED_STATE_FIELD_NUMBER: _ClassVar[int]
    allocation_id: str
    observed_state: str
    def __init__(self, allocation_id: _Optional[str] = ..., observed_state: _Optional[str] = ...) -> None: ...

class CompletionCapabilities(_message.Message):
    __slots__ = ("completion_contracts",)
    COMPLETION_CONTRACTS_FIELD_NUMBER: _ClassVar[int]
    completion_contracts: ListOfString
    def __init__(self, completion_contracts: _Optional[_Union[ListOfString, _Mapping]] = ...) -> None: ...

class NullableCompletionCapabilities(_message.Message):
    __slots__ = ("value", "null")
    VALUE_FIELD_NUMBER: _ClassVar[int]
    NULL_FIELD_NUMBER: _ClassVar[int]
    value: CompletionCapabilities
    null: _struct_pb2.NullValue
    def __init__(self, value: _Optional[_Union[CompletionCapabilities, _Mapping]] = ..., null: _Optional[_Union[_struct_pb2.NullValue, str]] = ...) -> None: ...

class WorkspaceCapabilities(_message.Message):
    __slots__ = ("storage", "modes", "limits")
    STORAGE_FIELD_NUMBER: _ClassVar[int]
    MODES_FIELD_NUMBER: _ClassVar[int]
    LIMITS_FIELD_NUMBER: _ClassVar[int]
    storage: str
    modes: ListOfString
    limits: WorkspaceLimits
    def __init__(self, storage: _Optional[str] = ..., modes: _Optional[_Union[ListOfString, _Mapping]] = ..., limits: _Optional[_Union[WorkspaceLimits, _Mapping]] = ...) -> None: ...

class WorkspaceLimits(_message.Message):
    __slots__ = ("max_files", "max_expanded_bytes", "max_managed_text_bytes", "max_file_bytes")
    MAX_FILES_FIELD_NUMBER: _ClassVar[int]
    MAX_EXPANDED_BYTES_FIELD_NUMBER: _ClassVar[int]
    MAX_MANAGED_TEXT_BYTES_FIELD_NUMBER: _ClassVar[int]
    MAX_FILE_BYTES_FIELD_NUMBER: _ClassVar[int]
    max_files: int
    max_expanded_bytes: int
    max_managed_text_bytes: int
    max_file_bytes: int
    def __init__(self, max_files: _Optional[int] = ..., max_expanded_bytes: _Optional[int] = ..., max_managed_text_bytes: _Optional[int] = ..., max_file_bytes: _Optional[int] = ...) -> None: ...

class NullableWorkspaceCapabilities(_message.Message):
    __slots__ = ("value", "null")
    VALUE_FIELD_NUMBER: _ClassVar[int]
    NULL_FIELD_NUMBER: _ClassVar[int]
    value: WorkspaceCapabilities
    null: _struct_pb2.NullValue
    def __init__(self, value: _Optional[_Union[WorkspaceCapabilities, _Mapping]] = ..., null: _Optional[_Union[_struct_pb2.NullValue, str]] = ...) -> None: ...

class ErrorDetail(_message.Message):
    __slots__ = ("code", "retryable", "request_id")
    CODE_FIELD_NUMBER: _ClassVar[int]
    RETRYABLE_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    code: str
    retryable: bool
    request_id: str
    def __init__(self, code: _Optional[str] = ..., retryable: _Optional[bool] = ..., request_id: _Optional[str] = ...) -> None: ...

class StateReadRequest(_message.Message):
    __slots__ = ("allocation_id", "if_revision")
    ALLOCATION_ID_FIELD_NUMBER: _ClassVar[int]
    IF_REVISION_FIELD_NUMBER: _ClassVar[int]
    allocation_id: str
    if_revision: int
    def __init__(self, allocation_id: _Optional[str] = ..., if_revision: _Optional[int] = ...) -> None: ...

class StateReadResponse(_message.Message):
    __slots__ = ("revision", "snapshot", "not_modified")
    REVISION_FIELD_NUMBER: _ClassVar[int]
    SNAPSHOT_FIELD_NUMBER: _ClassVar[int]
    NOT_MODIFIED_FIELD_NUMBER: _ClassVar[int]
    revision: int
    snapshot: AgentStateDocument
    not_modified: StateNotModified
    def __init__(self, revision: _Optional[int] = ..., snapshot: _Optional[_Union[AgentStateDocument, _Mapping]] = ..., not_modified: _Optional[_Union[StateNotModified, _Mapping]] = ...) -> None: ...

class StateNotModified(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class AgentStateDocument(_message.Message):
    __slots__ = ("json",)
    JSON_FIELD_NUMBER: _ClassVar[int]
    json: bytes
    def __init__(self, json: _Optional[bytes] = ...) -> None: ...

class PrepareAllocationRequest(_message.Message):
    __slots__ = ("api_version", "spec")
    API_VERSION_FIELD_NUMBER: _ClassVar[int]
    SPEC_FIELD_NUMBER: _ClassVar[int]
    api_version: str
    spec: AllocationSpec
    def __init__(self, api_version: _Optional[str] = ..., spec: _Optional[_Union[AllocationSpec, _Mapping]] = ...) -> None: ...

class PrepareAllocationResponse(_message.Message):
    __slots__ = ("api_version", "worker_handle")
    API_VERSION_FIELD_NUMBER: _ClassVar[int]
    WORKER_HANDLE_FIELD_NUMBER: _ClassVar[int]
    api_version: str
    worker_handle: WorkerHandle
    def __init__(self, api_version: _Optional[str] = ..., worker_handle: _Optional[_Union[WorkerHandle, _Mapping]] = ...) -> None: ...

class FinalizeAllocationRequest(_message.Message):
    __slots__ = ("api_version", "allocation_id", "finalization_id", "deadline")
    API_VERSION_FIELD_NUMBER: _ClassVar[int]
    ALLOCATION_ID_FIELD_NUMBER: _ClassVar[int]
    FINALIZATION_ID_FIELD_NUMBER: _ClassVar[int]
    DEADLINE_FIELD_NUMBER: _ClassVar[int]
    api_version: str
    allocation_id: str
    finalization_id: str
    deadline: str
    def __init__(self, api_version: _Optional[str] = ..., allocation_id: _Optional[str] = ..., finalization_id: _Optional[str] = ..., deadline: _Optional[str] = ...) -> None: ...

class AbortAllocationRequest(_message.Message):
    __slots__ = ("api_version", "allocation_id", "abort_id", "reason", "deadline")
    API_VERSION_FIELD_NUMBER: _ClassVar[int]
    ALLOCATION_ID_FIELD_NUMBER: _ClassVar[int]
    ABORT_ID_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    DEADLINE_FIELD_NUMBER: _ClassVar[int]
    api_version: str
    allocation_id: str
    abort_id: str
    reason: TerminationError
    deadline: str
    def __init__(self, api_version: _Optional[str] = ..., allocation_id: _Optional[str] = ..., abort_id: _Optional[str] = ..., reason: _Optional[_Union[TerminationError, _Mapping]] = ..., deadline: _Optional[str] = ...) -> None: ...

class ReleaseAllocationRequest(_message.Message):
    __slots__ = ("api_version", "allocation_id")
    API_VERSION_FIELD_NUMBER: _ClassVar[int]
    ALLOCATION_ID_FIELD_NUMBER: _ClassVar[int]
    api_version: str
    allocation_id: str
    def __init__(self, api_version: _Optional[str] = ..., allocation_id: _Optional[str] = ...) -> None: ...

class ReleaseAllocationResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class TerminationError(_message.Message):
    __slots__ = ("code", "message", "retryable")
    CODE_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    RETRYABLE_FIELD_NUMBER: _ClassVar[int]
    code: str
    message: str
    retryable: bool
    def __init__(self, code: _Optional[str] = ..., message: _Optional[str] = ..., retryable: _Optional[bool] = ...) -> None: ...

class AllocationSpec(_message.Message):
    __slots__ = ("api_version", "allocation_id", "run_id", "stage_execution_id", "logical_agent_name", "namespace", "worker_session_mode", "context_management", "run_metadata_labels", "project_id", "lease_expires_at", "agent_template", "resolved_skills", "model_policy", "summarizer_model_policy", "compactor_model_policy", "runtime_settings", "resolved_runtime_config_provenance", "workspace", "performance_metrics", "completion_contract", "engagement")
    API_VERSION_FIELD_NUMBER: _ClassVar[int]
    ALLOCATION_ID_FIELD_NUMBER: _ClassVar[int]
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    STAGE_EXECUTION_ID_FIELD_NUMBER: _ClassVar[int]
    LOGICAL_AGENT_NAME_FIELD_NUMBER: _ClassVar[int]
    NAMESPACE_FIELD_NUMBER: _ClassVar[int]
    WORKER_SESSION_MODE_FIELD_NUMBER: _ClassVar[int]
    CONTEXT_MANAGEMENT_FIELD_NUMBER: _ClassVar[int]
    RUN_METADATA_LABELS_FIELD_NUMBER: _ClassVar[int]
    PROJECT_ID_FIELD_NUMBER: _ClassVar[int]
    LEASE_EXPIRES_AT_FIELD_NUMBER: _ClassVar[int]
    AGENT_TEMPLATE_FIELD_NUMBER: _ClassVar[int]
    RESOLVED_SKILLS_FIELD_NUMBER: _ClassVar[int]
    MODEL_POLICY_FIELD_NUMBER: _ClassVar[int]
    SUMMARIZER_MODEL_POLICY_FIELD_NUMBER: _ClassVar[int]
    COMPACTOR_MODEL_POLICY_FIELD_NUMBER: _ClassVar[int]
    RUNTIME_SETTINGS_FIELD_NUMBER: _ClassVar[int]
    RESOLVED_RUNTIME_CONFIG_PROVENANCE_FIELD_NUMBER: _ClassVar[int]
    WORKSPACE_FIELD_NUMBER: _ClassVar[int]
    PERFORMANCE_METRICS_FIELD_NUMBER: _ClassVar[int]
    COMPLETION_CONTRACT_FIELD_NUMBER: _ClassVar[int]
    ENGAGEMENT_FIELD_NUMBER: _ClassVar[int]
    api_version: str
    allocation_id: str
    run_id: str
    stage_execution_id: str
    logical_agent_name: str
    namespace: str
    worker_session_mode: str
    context_management: WorkerContextManagement
    run_metadata_labels: MapOfString
    project_id: str
    lease_expires_at: str
    agent_template: AgentTemplateDocument
    resolved_skills: ListOfResolvedSkill
    model_policy: ModelPolicyDocument
    summarizer_model_policy: ModelPolicyDocument
    compactor_model_policy: ModelPolicyDocument
    runtime_settings: RuntimeSettingsDocument
    resolved_runtime_config_provenance: RuntimeConfigProvenanceDocument
    workspace: NullableAllocationWorkspace
    performance_metrics: NullablePerformanceMetricsRequest
    completion_contract: WorkerCompletionContract
    engagement: EngagementAllocation
    def __init__(self, api_version: _Optional[str] = ..., allocation_id: _Optional[str] = ..., run_id: _Optional[str] = ..., stage_execution_id: _Optional[str] = ..., logical_agent_name: _Optional[str] = ..., namespace: _Optional[str] = ..., worker_session_mode: _Optional[str] = ..., context_management: _Optional[_Union[WorkerContextManagement, _Mapping]] = ..., run_metadata_labels: _Optional[_Union[MapOfString, _Mapping]] = ..., project_id: _Optional[str] = ..., lease_expires_at: _Optional[str] = ..., agent_template: _Optional[_Union[AgentTemplateDocument, _Mapping]] = ..., resolved_skills: _Optional[_Union[ListOfResolvedSkill, _Mapping]] = ..., model_policy: _Optional[_Union[ModelPolicyDocument, _Mapping]] = ..., summarizer_model_policy: _Optional[_Union[ModelPolicyDocument, _Mapping]] = ..., compactor_model_policy: _Optional[_Union[ModelPolicyDocument, _Mapping]] = ..., runtime_settings: _Optional[_Union[RuntimeSettingsDocument, _Mapping]] = ..., resolved_runtime_config_provenance: _Optional[_Union[RuntimeConfigProvenanceDocument, _Mapping]] = ..., workspace: _Optional[_Union[NullableAllocationWorkspace, _Mapping]] = ..., performance_metrics: _Optional[_Union[NullablePerformanceMetricsRequest, _Mapping]] = ..., completion_contract: _Optional[_Union[WorkerCompletionContract, _Mapping]] = ..., engagement: _Optional[_Union[EngagementAllocation, _Mapping]] = ...) -> None: ...

class AgentTemplateDocument(_message.Message):
    __slots__ = ("json",)
    JSON_FIELD_NUMBER: _ClassVar[int]
    json: bytes
    def __init__(self, json: _Optional[bytes] = ...) -> None: ...

class ModelPolicyDocument(_message.Message):
    __slots__ = ("json",)
    JSON_FIELD_NUMBER: _ClassVar[int]
    json: bytes
    def __init__(self, json: _Optional[bytes] = ...) -> None: ...

class RuntimeSettingsDocument(_message.Message):
    __slots__ = ("json",)
    JSON_FIELD_NUMBER: _ClassVar[int]
    json: bytes
    def __init__(self, json: _Optional[bytes] = ...) -> None: ...

class RuntimeConfigProvenanceDocument(_message.Message):
    __slots__ = ("json",)
    JSON_FIELD_NUMBER: _ClassVar[int]
    json: bytes
    def __init__(self, json: _Optional[bytes] = ...) -> None: ...

class AgentCardDocument(_message.Message):
    __slots__ = ("json",)
    JSON_FIELD_NUMBER: _ClassVar[int]
    json: bytes
    def __init__(self, json: _Optional[bytes] = ...) -> None: ...

class ExecutionReportDocument(_message.Message):
    __slots__ = ("json",)
    JSON_FIELD_NUMBER: _ClassVar[int]
    json: bytes
    def __init__(self, json: _Optional[bytes] = ...) -> None: ...

class RuntimeReportDocument(_message.Message):
    __slots__ = ("json",)
    JSON_FIELD_NUMBER: _ClassVar[int]
    json: bytes
    def __init__(self, json: _Optional[bytes] = ...) -> None: ...

class MapOfString(_message.Message):
    __slots__ = ("values",)
    class ValuesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    VALUES_FIELD_NUMBER: _ClassVar[int]
    values: _containers.ScalarMap[str, str]
    def __init__(self, values: _Optional[_Mapping[str, str]] = ...) -> None: ...

class ResolvedInstructions(_message.Message):
    __slots__ = ("ref", "digest", "text")
    REF_FIELD_NUMBER: _ClassVar[int]
    DIGEST_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    ref: str
    digest: str
    text: str
    def __init__(self, ref: _Optional[str] = ..., digest: _Optional[str] = ..., text: _Optional[str] = ...) -> None: ...

class WorkerContextManagement(_message.Message):
    __slots__ = ("strategy", "instructions", "include_reasoning", "context_window_ratio", "keep_recent_groups", "max_compactions")
    STRATEGY_FIELD_NUMBER: _ClassVar[int]
    INSTRUCTIONS_FIELD_NUMBER: _ClassVar[int]
    INCLUDE_REASONING_FIELD_NUMBER: _ClassVar[int]
    CONTEXT_WINDOW_RATIO_FIELD_NUMBER: _ClassVar[int]
    KEEP_RECENT_GROUPS_FIELD_NUMBER: _ClassVar[int]
    MAX_COMPACTIONS_FIELD_NUMBER: _ClassVar[int]
    strategy: str
    instructions: ResolvedInstructions
    include_reasoning: bool
    context_window_ratio: float
    keep_recent_groups: int
    max_compactions: int
    def __init__(self, strategy: _Optional[str] = ..., instructions: _Optional[_Union[ResolvedInstructions, _Mapping]] = ..., include_reasoning: _Optional[bool] = ..., context_window_ratio: _Optional[float] = ..., keep_recent_groups: _Optional[int] = ..., max_compactions: _Optional[int] = ...) -> None: ...

class ArtifactRef(_message.Message):
    __slots__ = ("namespace", "name", "revision")
    NAMESPACE_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    REVISION_FIELD_NUMBER: _ClassVar[int]
    namespace: str
    name: str
    revision: str
    def __init__(self, namespace: _Optional[str] = ..., name: _Optional[str] = ..., revision: _Optional[str] = ...) -> None: ...

class ResolvedSkill(_message.Message):
    __slots__ = ("name", "artifact", "package_digest")
    NAME_FIELD_NUMBER: _ClassVar[int]
    ARTIFACT_FIELD_NUMBER: _ClassVar[int]
    PACKAGE_DIGEST_FIELD_NUMBER: _ClassVar[int]
    name: str
    artifact: ArtifactRef
    package_digest: str
    def __init__(self, name: _Optional[str] = ..., artifact: _Optional[_Union[ArtifactRef, _Mapping]] = ..., package_digest: _Optional[str] = ...) -> None: ...

class ListOfResolvedSkill(_message.Message):
    __slots__ = ("values",)
    VALUES_FIELD_NUMBER: _ClassVar[int]
    values: _containers.RepeatedCompositeFieldContainer[ResolvedSkill]
    def __init__(self, values: _Optional[_Iterable[_Union[ResolvedSkill, _Mapping]]] = ...) -> None: ...

class WorkerCompletionContract(_message.Message):
    __slots__ = ("kind", "context", "result_artifact")
    KIND_FIELD_NUMBER: _ClassVar[int]
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    RESULT_ARTIFACT_FIELD_NUMBER: _ClassVar[int]
    kind: str
    context: ArtifactRef
    result_artifact: ArtifactRef
    def __init__(self, kind: _Optional[str] = ..., context: _Optional[_Union[ArtifactRef, _Mapping]] = ..., result_artifact: _Optional[_Union[ArtifactRef, _Mapping]] = ...) -> None: ...

class AgentTemplateRef(_message.Message):
    __slots__ = ("template_id", "version", "digest")
    TEMPLATE_ID_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    DIGEST_FIELD_NUMBER: _ClassVar[int]
    template_id: str
    version: str
    digest: str
    def __init__(self, template_id: _Optional[str] = ..., version: _Optional[str] = ..., digest: _Optional[str] = ...) -> None: ...

class WorkerRuntimeRef(_message.Message):
    __slots__ = ("runtime_id", "version")
    RUNTIME_ID_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    runtime_id: str
    version: str
    def __init__(self, runtime_id: _Optional[str] = ..., version: _Optional[str] = ...) -> None: ...

class WorkerHandle(_message.Message):
    __slots__ = ("allocation_id", "agent_template_ref", "worker_runtime_ref", "agent_card", "lease_expires_at")
    ALLOCATION_ID_FIELD_NUMBER: _ClassVar[int]
    AGENT_TEMPLATE_REF_FIELD_NUMBER: _ClassVar[int]
    WORKER_RUNTIME_REF_FIELD_NUMBER: _ClassVar[int]
    AGENT_CARD_FIELD_NUMBER: _ClassVar[int]
    LEASE_EXPIRES_AT_FIELD_NUMBER: _ClassVar[int]
    allocation_id: str
    agent_template_ref: AgentTemplateRef
    worker_runtime_ref: WorkerRuntimeRef
    agent_card: AgentCardDocument
    lease_expires_at: str
    def __init__(self, allocation_id: _Optional[str] = ..., agent_template_ref: _Optional[_Union[AgentTemplateRef, _Mapping]] = ..., worker_runtime_ref: _Optional[_Union[WorkerRuntimeRef, _Mapping]] = ..., agent_card: _Optional[_Union[AgentCardDocument, _Mapping]] = ..., lease_expires_at: _Optional[str] = ...) -> None: ...

class AllocationWorkspace(_message.Message):
    __slots__ = ("mode", "sources", "state", "export")
    MODE_FIELD_NUMBER: _ClassVar[int]
    SOURCES_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    EXPORT_FIELD_NUMBER: _ClassVar[int]
    mode: str
    sources: ListOfWorkspaceSource
    state: NullableWorkspaceState
    export: NullableWorkspaceExport
    def __init__(self, mode: _Optional[str] = ..., sources: _Optional[_Union[ListOfWorkspaceSource, _Mapping]] = ..., state: _Optional[_Union[NullableWorkspaceState, _Mapping]] = ..., export: _Optional[_Union[NullableWorkspaceExport, _Mapping]] = ...) -> None: ...

class WorkspaceSource(_message.Message):
    __slots__ = ("artifact", "target")
    ARTIFACT_FIELD_NUMBER: _ClassVar[int]
    TARGET_FIELD_NUMBER: _ClassVar[int]
    artifact: ArtifactRef
    target: str
    def __init__(self, artifact: _Optional[_Union[ArtifactRef, _Mapping]] = ..., target: _Optional[str] = ...) -> None: ...

class ListOfWorkspaceSource(_message.Message):
    __slots__ = ("values",)
    VALUES_FIELD_NUMBER: _ClassVar[int]
    values: _containers.RepeatedCompositeFieldContainer[WorkspaceSource]
    def __init__(self, values: _Optional[_Iterable[_Union[WorkspaceSource, _Mapping]]] = ...) -> None: ...

class WorkspaceState(_message.Message):
    __slots__ = ("artifact",)
    ARTIFACT_FIELD_NUMBER: _ClassVar[int]
    artifact: ArtifactRef
    def __init__(self, artifact: _Optional[_Union[ArtifactRef, _Mapping]] = ...) -> None: ...

class WorkspaceExport(_message.Message):
    __slots__ = ("state", "diff")
    STATE_FIELD_NUMBER: _ClassVar[int]
    DIFF_FIELD_NUMBER: _ClassVar[int]
    state: str
    diff: str
    def __init__(self, state: _Optional[str] = ..., diff: _Optional[str] = ...) -> None: ...

class NullableAllocationWorkspace(_message.Message):
    __slots__ = ("value", "null")
    VALUE_FIELD_NUMBER: _ClassVar[int]
    NULL_FIELD_NUMBER: _ClassVar[int]
    value: AllocationWorkspace
    null: _struct_pb2.NullValue
    def __init__(self, value: _Optional[_Union[AllocationWorkspace, _Mapping]] = ..., null: _Optional[_Union[_struct_pb2.NullValue, str]] = ...) -> None: ...

class NullableWorkspaceState(_message.Message):
    __slots__ = ("value", "null")
    VALUE_FIELD_NUMBER: _ClassVar[int]
    NULL_FIELD_NUMBER: _ClassVar[int]
    value: WorkspaceState
    null: _struct_pb2.NullValue
    def __init__(self, value: _Optional[_Union[WorkspaceState, _Mapping]] = ..., null: _Optional[_Union[_struct_pb2.NullValue, str]] = ...) -> None: ...

class NullableWorkspaceExport(_message.Message):
    __slots__ = ("value", "null")
    VALUE_FIELD_NUMBER: _ClassVar[int]
    NULL_FIELD_NUMBER: _ClassVar[int]
    value: WorkspaceExport
    null: _struct_pb2.NullValue
    def __init__(self, value: _Optional[_Union[WorkspaceExport, _Mapping]] = ..., null: _Optional[_Union[_struct_pb2.NullValue, str]] = ...) -> None: ...

class PerformanceMetricsRequest(_message.Message):
    __slots__ = ("version", "interval_seconds")
    VERSION_FIELD_NUMBER: _ClassVar[int]
    INTERVAL_SECONDS_FIELD_NUMBER: _ClassVar[int]
    version: int
    interval_seconds: int
    def __init__(self, version: _Optional[int] = ..., interval_seconds: _Optional[int] = ...) -> None: ...

class NullablePerformanceMetricsRequest(_message.Message):
    __slots__ = ("value", "null")
    VALUE_FIELD_NUMBER: _ClassVar[int]
    NULL_FIELD_NUMBER: _ClassVar[int]
    value: PerformanceMetricsRequest
    null: _struct_pb2.NullValue
    def __init__(self, value: _Optional[_Union[PerformanceMetricsRequest, _Mapping]] = ..., null: _Optional[_Union[_struct_pb2.NullValue, str]] = ...) -> None: ...

class EngagementAllocation(_message.Message):
    __slots__ = ("schema", "engagement", "engagement_digest", "authorized_by", "authorized_at", "http")
    SCHEMA_FIELD_NUMBER: _ClassVar[int]
    ENGAGEMENT_FIELD_NUMBER: _ClassVar[int]
    ENGAGEMENT_DIGEST_FIELD_NUMBER: _ClassVar[int]
    AUTHORIZED_BY_FIELD_NUMBER: _ClassVar[int]
    AUTHORIZED_AT_FIELD_NUMBER: _ClassVar[int]
    HTTP_FIELD_NUMBER: _ClassVar[int]
    schema: str
    engagement: ArtifactRef
    engagement_digest: str
    authorized_by: str
    authorized_at: str
    http: HTTPAssignment
    def __init__(self, schema: _Optional[str] = ..., engagement: _Optional[_Union[ArtifactRef, _Mapping]] = ..., engagement_digest: _Optional[str] = ..., authorized_by: _Optional[str] = ..., authorized_at: _Optional[str] = ..., http: _Optional[_Union[HTTPAssignment, _Mapping]] = ...) -> None: ...

class HTTPAssignment(_message.Message):
    __slots__ = ("limits", "adapter", "routes", "exclusions", "identities", "protected_destinations")
    LIMITS_FIELD_NUMBER: _ClassVar[int]
    ADAPTER_FIELD_NUMBER: _ClassVar[int]
    ROUTES_FIELD_NUMBER: _ClassVar[int]
    EXCLUSIONS_FIELD_NUMBER: _ClassVar[int]
    IDENTITIES_FIELD_NUMBER: _ClassVar[int]
    PROTECTED_DESTINATIONS_FIELD_NUMBER: _ClassVar[int]
    limits: HTTPLimits
    adapter: str
    routes: ListOfHTTPRoute
    exclusions: ListOfHTTPRoute
    identities: ListOfHTTPIdentity
    protected_destinations: ListOfProtectedDestination
    def __init__(self, limits: _Optional[_Union[HTTPLimits, _Mapping]] = ..., adapter: _Optional[str] = ..., routes: _Optional[_Union[ListOfHTTPRoute, _Mapping]] = ..., exclusions: _Optional[_Union[ListOfHTTPRoute, _Mapping]] = ..., identities: _Optional[_Union[ListOfHTTPIdentity, _Mapping]] = ..., protected_destinations: _Optional[_Union[ListOfProtectedDestination, _Mapping]] = ...) -> None: ...

class HTTPLimits(_message.Message):
    __slots__ = ("requests_per_second", "concurrent_requests", "request_body_bytes", "response_body_bytes")
    REQUESTS_PER_SECOND_FIELD_NUMBER: _ClassVar[int]
    CONCURRENT_REQUESTS_FIELD_NUMBER: _ClassVar[int]
    REQUEST_BODY_BYTES_FIELD_NUMBER: _ClassVar[int]
    RESPONSE_BODY_BYTES_FIELD_NUMBER: _ClassVar[int]
    requests_per_second: int
    concurrent_requests: int
    request_body_bytes: int
    response_body_bytes: int
    def __init__(self, requests_per_second: _Optional[int] = ..., concurrent_requests: _Optional[int] = ..., request_body_bytes: _Optional[int] = ..., response_body_bytes: _Optional[int] = ...) -> None: ...

class HTTPRoute(_message.Message):
    __slots__ = ("origin", "path_prefixes", "methods", "address_ranges")
    ORIGIN_FIELD_NUMBER: _ClassVar[int]
    PATH_PREFIXES_FIELD_NUMBER: _ClassVar[int]
    METHODS_FIELD_NUMBER: _ClassVar[int]
    ADDRESS_RANGES_FIELD_NUMBER: _ClassVar[int]
    origin: str
    path_prefixes: ListOfString
    methods: ListOfString
    address_ranges: ListOfString
    def __init__(self, origin: _Optional[str] = ..., path_prefixes: _Optional[_Union[ListOfString, _Mapping]] = ..., methods: _Optional[_Union[ListOfString, _Mapping]] = ..., address_ranges: _Optional[_Union[ListOfString, _Mapping]] = ...) -> None: ...

class HTTPIdentity(_message.Message):
    __slots__ = ("id", "kind", "origins", "role", "tenant", "credential")
    ID_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    ORIGINS_FIELD_NUMBER: _ClassVar[int]
    ROLE_FIELD_NUMBER: _ClassVar[int]
    TENANT_FIELD_NUMBER: _ClassVar[int]
    CREDENTIAL_FIELD_NUMBER: _ClassVar[int]
    id: str
    kind: str
    origins: ListOfString
    role: str
    tenant: str
    credential: RuntimeCredentialRef
    def __init__(self, id: _Optional[str] = ..., kind: _Optional[str] = ..., origins: _Optional[_Union[ListOfString, _Mapping]] = ..., role: _Optional[str] = ..., tenant: _Optional[str] = ..., credential: _Optional[_Union[RuntimeCredentialRef, _Mapping]] = ...) -> None: ...

class RuntimeCredentialRef(_message.Message):
    __slots__ = ("credential_id", "kind")
    CREDENTIAL_ID_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    credential_id: str
    kind: str
    def __init__(self, credential_id: _Optional[str] = ..., kind: _Optional[str] = ...) -> None: ...

class ProtectedDestination(_message.Message):
    __slots__ = ("hostname", "addresses", "port")
    HOSTNAME_FIELD_NUMBER: _ClassVar[int]
    ADDRESSES_FIELD_NUMBER: _ClassVar[int]
    PORT_FIELD_NUMBER: _ClassVar[int]
    hostname: str
    addresses: ListOfString
    port: int
    def __init__(self, hostname: _Optional[str] = ..., addresses: _Optional[_Union[ListOfString, _Mapping]] = ..., port: _Optional[int] = ...) -> None: ...

class ListOfHTTPRoute(_message.Message):
    __slots__ = ("values",)
    VALUES_FIELD_NUMBER: _ClassVar[int]
    values: _containers.RepeatedCompositeFieldContainer[HTTPRoute]
    def __init__(self, values: _Optional[_Iterable[_Union[HTTPRoute, _Mapping]]] = ...) -> None: ...

class ListOfHTTPIdentity(_message.Message):
    __slots__ = ("values",)
    VALUES_FIELD_NUMBER: _ClassVar[int]
    values: _containers.RepeatedCompositeFieldContainer[HTTPIdentity]
    def __init__(self, values: _Optional[_Iterable[_Union[HTTPIdentity, _Mapping]]] = ...) -> None: ...

class ListOfProtectedDestination(_message.Message):
    __slots__ = ("values",)
    VALUES_FIELD_NUMBER: _ClassVar[int]
    values: _containers.RepeatedCompositeFieldContainer[ProtectedDestination]
    def __init__(self, values: _Optional[_Iterable[_Union[ProtectedDestination, _Mapping]]] = ...) -> None: ...

class AllocationFinalResponse(_message.Message):
    __slots__ = ("api_version", "report")
    API_VERSION_FIELD_NUMBER: _ClassVar[int]
    REPORT_FIELD_NUMBER: _ClassVar[int]
    api_version: str
    report: AllocationFinalReport
    def __init__(self, api_version: _Optional[str] = ..., report: _Optional[_Union[AllocationFinalReport, _Mapping]] = ...) -> None: ...

class AllocationFinalReport(_message.Message):
    __slots__ = ("report_id", "allocation_id", "started_at", "finished_at", "worker", "runtime")
    REPORT_ID_FIELD_NUMBER: _ClassVar[int]
    ALLOCATION_ID_FIELD_NUMBER: _ClassVar[int]
    STARTED_AT_FIELD_NUMBER: _ClassVar[int]
    FINISHED_AT_FIELD_NUMBER: _ClassVar[int]
    WORKER_FIELD_NUMBER: _ClassVar[int]
    RUNTIME_FIELD_NUMBER: _ClassVar[int]
    report_id: str
    allocation_id: str
    started_at: str
    finished_at: str
    worker: ExecutionReportDocument
    runtime: RuntimeReportDocument
    def __init__(self, report_id: _Optional[str] = ..., allocation_id: _Optional[str] = ..., started_at: _Optional[str] = ..., finished_at: _Optional[str] = ..., worker: _Optional[_Union[ExecutionReportDocument, _Mapping]] = ..., runtime: _Optional[_Union[RuntimeReportDocument, _Mapping]] = ...) -> None: ...
