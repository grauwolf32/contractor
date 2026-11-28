import type {
  AgentTemplateBody,
  ConfigurationResource,
  LLMGatewayBody,
  ModelPolicyBody,
} from "../../../api/operations";
import { Link } from "react-router";
import { formatNumber } from "../../../app/format";
import { Facts } from "../../../ui";
import { artifactDetailPath } from "../../../app/artifact-path";

function present(value: number | undefined): string {
  return value === undefined ? "omitted" : formatNumber(value);
}

function ModelPolicyView({ body }: { body: ModelPolicyBody }) {
  const fields = [
    ["Model alias", body.model],
    [
      "Image input",
      body.supportsImages === true
        ? "Supported"
        : body.supportsImages === false
          ? "Text only"
          : "Unconfirmed",
    ],
    ["Context window tokens", present(body.contextWindowTokens)],
    ["Maximum output tokens", present(body.maxOutputTokens)],
    ["Maximum model calls", present(body.maxModelCalls)],
    ["Maximum tool calls", present(body.maxToolCalls)],
    ["Maximum Worker calls", present(body.maxWorkerCalls)],
    ["Maximum total tokens", present(body.maxTotalTokens)],
    ["Temperature", present(body.temperature)],
  ] as const;
  return (
    <Facts
      layout="rows"
      className="configuration-body"
      items={fields.map(([term, value]) => ({
        term,
        value: <code>{value}</code>,
      }))}
    />
  );
}

function GatewayView({ body }: { body: LLMGatewayBody }) {
  return (
    <Facts
      layout="rows"
      className="configuration-body"
      items={[
        { term: "Protocol", value: <code>{body.protocol}</code> },
        { term: "Inference URL", value: <code>{body.url}</code> },
        {
          term: "Credential manager",
          value:
            body.credentialManager === undefined ? (
              <span className="ops-muted">Not configured</span>
            ) : (
              <span className="ops-value-stack">
                <code>{body.credentialManager.implementation}</code>
                <code>{body.credentialManager.managementUrl}</code>
              </span>
            ),
        },
        {
          term: "Failure signatures",
          value:
            body.failureSignatures === undefined ? (
              <span className="ops-muted">Protocol default</span>
            ) : (
              <span className="ops-value-stack">
                {(body.failureSignatures.modelUnavailable ?? []).map(
                  (signature, index) => (
                    <code key={`unavailable-${index}`}>
                      {signature.status} →{" "}
                      {signature.messageEquals ?? signature.litellmWrapped}
                      {signature.litellmWrapped === undefined
                        ? ""
                        : " (LiteLLM wrapped)"}
                    </code>
                  ),
                )}
                {(body.failureSignatures.permanentCodes ?? []).length ===
                0 ? null : (
                  <code>
                    permanent:{" "}
                    {body.failureSignatures.permanentCodes?.join(", ")}
                  </code>
                )}
              </span>
            ),
        },
      ]}
    />
  );
}

function AgentTemplateView({ body }: { body: AgentTemplateBody }) {
  return (
    <Facts
      layout="rows"
      className="configuration-body"
      items={[
        { term: "Description", value: body.description },
        { term: "Runtime implementation", value: <code>{body.runtime}</code> },
        {
          term: "Instructions",
          value: body.instructions && (
            <span className="ops-value-stack">
              <code>{body.instructions.ref}</code>
              <code>{body.instructions.digest}</code>
            </span>
          ),
        },
        {
          term: "Tool execution",
          value: body.execution && (
            <>
              <code>{body.execution.tool}</code> ·{" "}
              {body.execution.timeoutSeconds}s
            </>
          ),
        },
        { term: "Sandbox profile", value: <code>{body.sandboxProfile}</code> },
        {
          term: "Toolset selections",
          value:
            body.toolsets.length === 0 ? (
              <span className="ops-muted">No model-visible tools</span>
            ) : (
              <ul className="ops-value-list">
                {body.toolsets.map((selection) => (
                  <li key={selection.ref}>
                    <code>{selection.ref}</code> · {selection.tools.join(", ")}
                  </li>
                ))}
              </ul>
            ),
        },
        {
          term: "Global Skills",
          value:
            body.skills === undefined || body.skills.length === 0 ? (
              <span className="ops-muted">None</span>
            ) : (
              <ul className="ops-value-list">
                {body.skills.map((skill) => (
                  <li key={`${skill.namespace}/${skill.name}`}>
                    <Link
                      to={artifactDetailPath(
                        { kind: "user" },
                        { namespace: skill.namespace, name: skill.name },
                      )}
                    >
                      <code>
                        {skill.namespace}/{skill.name}
                      </code>
                    </Link>
                  </li>
                ))}
              </ul>
            ),
        },
      ]}
    />
  );
}

function ConfigurationBody({ resource }: { resource: ConfigurationResource }) {
  switch (resource.ref.kind) {
    case "model-policies":
      return <ModelPolicyView body={resource.body as ModelPolicyBody} />;
    case "llm-gateways":
      return <GatewayView body={resource.body as LLMGatewayBody} />;
    case "agent-templates":
      return <AgentTemplateView body={resource.body as AgentTemplateBody} />;
  }
}

/**
 * The published values of a configuration as label / value facts. The frame
 * lays them out by the width it gets, not by the viewport, so the same view
 * fits an Operations page and the Library agent page's side panel. The
 * `configuration-body` class on the list is a hook for other areas.
 */
export function ConfigurationBodyView({
  resource,
}: {
  resource: ConfigurationResource;
}) {
  return (
    <div className="ui-facts-frame">
      <ConfigurationBody resource={resource} />
    </div>
  );
}
