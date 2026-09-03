import type {
  AgentTemplateBody,
  ConfigurationResource,
  ExecutionConfigBody,
  LLMGatewayBody,
  ModelPolicyBody,
} from "../../../api/operations";
import { ConfigurationRefLink } from "../common";
import { exactConfigurationRef } from "../references";

type ReferenceSummary = { description?: string; refs?: string[] };

function ReferenceSummaryView({ body }: { body: ReferenceSummary }) {
  return (
    <dl className="key-value-list configuration-body">
      <div>
        <dt>Description</dt>
        <dd>
          {body.description ?? <span className="muted-copy">Not supplied</span>}
        </dd>
      </div>
      <div>
        <dt>Resolved references</dt>
        <dd>
          {body.refs === undefined || body.refs.length === 0 ? (
            <span className="muted-copy">No reference summary</span>
          ) : (
            <ul className="compact-value-list">
              {body.refs.map((ref) => (
                <li key={ref}>
                  <code>{ref}</code>
                </li>
              ))}
            </ul>
          )}
        </dd>
      </div>
    </dl>
  );
}

function present(value: number | undefined): string {
  return value === undefined ? "omitted" : value.toLocaleString();
}

function ModelPolicyView({ body }: { body: ModelPolicyBody }) {
  const fields = [
    ["Model alias", body.model],
    ["Context window tokens", present(body.contextWindowTokens)],
    ["Maximum output tokens", present(body.maxOutputTokens)],
    ["Maximum model calls", present(body.maxModelCalls)],
    ["Maximum tool calls", present(body.maxToolCalls)],
    ["Maximum Worker calls", present(body.maxWorkerCalls)],
    ["Maximum total tokens", present(body.maxTotalTokens)],
    ["Temperature", present(body.temperature)],
  ] as const;
  return (
    <dl className="key-value-list configuration-body">
      {fields.map(([label, value]) => (
        <div key={label}>
          <dt>{label}</dt>
          <dd>
            <code>{value}</code>
          </dd>
        </div>
      ))}
    </dl>
  );
}

function GatewayView({ body }: { body: LLMGatewayBody }) {
  return (
    <dl className="key-value-list configuration-body">
      <div>
        <dt>Protocol</dt>
        <dd>
          <code>{body.protocol}</code>
        </dd>
      </div>
      <div>
        <dt>Inference URL</dt>
        <dd>
          <code>{body.url}</code>
        </dd>
      </div>
      <div>
        <dt>Credential manager</dt>
        <dd>
          {body.credentialManager === undefined ? (
            <span className="muted-copy">Not configured</span>
          ) : (
            <span className="nested-value">
              <code>{body.credentialManager.implementation}</code>
              <code>{body.credentialManager.managementUrl}</code>
            </span>
          )}
        </dd>
      </div>
    </dl>
  );
}

function AgentTemplateView({ body }: { body: AgentTemplateBody }) {
  return (
    <dl className="key-value-list configuration-body">
      <div>
        <dt>Description</dt>
        <dd>{body.description}</dd>
      </div>
      <div>
        <dt>Runtime implementation</dt>
        <dd>
          <code>{body.runtime}</code>
        </dd>
      </div>
      <div>
        <dt>Instructions</dt>
        <dd className="nested-value">
          <code>{body.instructions.ref}</code>
          <code>{body.instructions.digest}</code>
        </dd>
      </div>
      <div>
        <dt>Default ModelPolicy</dt>
        <dd>
          <ConfigurationRefLink
            value={exactConfigurationRef(body.modelPolicy)}
          />
        </dd>
      </div>
      <div>
        <dt>Sandbox profile</dt>
        <dd>
          <code>{body.sandboxProfile}</code>
        </dd>
      </div>
      <div>
        <dt>Toolset selections</dt>
        <dd>
          {body.toolsets.length === 0 ? (
            <span className="muted-copy">No model-visible tools</span>
          ) : (
            <ul className="compact-value-list">
              {body.toolsets.map((selection) => (
                <li key={selection.ref}>
                  <code>{selection.ref}</code> · {selection.tools.join(", ")}
                </li>
              ))}
            </ul>
          )}
        </dd>
      </div>
    </dl>
  );
}

function ExecutionSelection({
  selection,
}: {
  selection: NonNullable<ExecutionConfigBody["planner"]>;
}) {
  return (
    <ul className="compact-value-list">
      {selection.modelPolicy === undefined ? null : (
        <li>
          ModelPolicy:{" "}
          <ConfigurationRefLink
            value={exactConfigurationRef(selection.modelPolicy)}
          />
        </li>
      )}
      {selection.llmGateway === undefined ? null : (
        <li>
          Gateway:{" "}
          <ConfigurationRefLink
            value={exactConfigurationRef(selection.llmGateway)}
          />
        </li>
      )}
      {"credential" in selection ? (
        <li>
          Credential:{" "}
          <code>{selection.credential?.credentialId ?? "explicitly none"}</code>
        </li>
      ) : null}
    </ul>
  );
}

function ExecutionConfigView({ body }: { body: ExecutionConfigBody }) {
  return (
    <dl className="key-value-list configuration-body">
      <div>
        <dt>Planner overrides</dt>
        <dd>
          {body.planner === undefined ? (
            <span className="muted-copy">None</span>
          ) : (
            <ExecutionSelection selection={body.planner} />
          )}
        </dd>
      </div>
      {Object.entries(body.agents ?? {})
        .sort(([left], [right]) => left.localeCompare(right))
        .map(([name, selection]) => (
          <div key={name}>
            <dt>Agent {name}</dt>
            <dd>
              <ExecutionSelection selection={selection} />
            </dd>
          </div>
        ))}
    </dl>
  );
}

export function ConfigurationBodyView({
  resource,
}: {
  resource: ConfigurationResource;
}) {
  switch (resource.ref.kind) {
    case "model-policies":
      return <ModelPolicyView body={resource.body as ModelPolicyBody} />;
    case "llm-gateways":
      return <GatewayView body={resource.body as LLMGatewayBody} />;
    case "agent-templates":
      return "runtime" in resource.body ? (
        <AgentTemplateView body={resource.body as AgentTemplateBody} />
      ) : (
        <ReferenceSummaryView body={resource.body as ReferenceSummary} />
      );
    case "execution-configs":
      return "description" in resource.body || "refs" in resource.body ? (
        <ReferenceSummaryView body={resource.body as ReferenceSummary} />
      ) : (
        <ExecutionConfigView body={resource.body as ExecutionConfigBody} />
      );
  }
}
