import type { components } from "../../api/generated/public";
import { compactDigest } from "../../app/format";

type ConsumerConfig = components["schemas"]["ConsumerExecutionConfig"];
type ResolvedConfig = components["schemas"]["ResolvedStageExecutionConfig"];

function Digest({ value }: { value: string }) {
  return (
    <code className="ui-mono" title={value}>
      {compactDigest(value)}
    </code>
  );
}

/**
 * One consumer of a resolved stage configuration (the Planner or a Worker):
 * its model, Gateway and credential in the user's words. `exact` adds the
 * digests and the origin of each value, for a Run's technical details.
 */
function ConsumerConfigView({
  label,
  config,
  exact,
}: {
  label: string;
  config: ConsumerConfig;
  exact: boolean;
}) {
  if (config.modelPolicy === undefined) {
    return (
      <li>
        <strong>{label}</strong>
        <span>Tool execution · no model</span>
      </li>
    );
  }
  return (
    <li>
      <strong>{label}</strong>
      <span>
        Model{" "}
        <code>
          {config.modelPolicy.policyId}@{config.modelPolicy.version}
        </code>
        {exact ? (
          <>
            {" "}
            <Digest value={config.modelPolicy.digest} />
          </>
        ) : null}
      </span>
      {config.model === undefined ? null : (
        <span>
          Requested model <code>{config.model}</code>
        </span>
      )}
      {config.summarizerModel === undefined ? null : (
        <span>
          Summarizer <code>{config.summarizerModel}</code>
        </span>
      )}
      {config.llmGateway === undefined ? (
        <span>Gateway resolved during Runtime placement</span>
      ) : (
        <span>
          Gateway{" "}
          <code>
            {config.llmGateway.gatewayId}@{config.llmGateway.version}
          </code>
          {exact ? (
            <>
              {" "}
              <Digest value={config.llmGateway.digest} />
            </>
          ) : null}
        </span>
      )}
      <span>Credential {config.credential?.credentialId ?? "none"}</span>
      {exact ? (
        <span className="ui-caption">
          Origins: model {config.origins?.modelPolicy ?? "not reported"};
          Gateway {config.origins?.llmGateway ?? "not reported"}; credential{" "}
          {config.origins?.credential ?? "not reported"}
        </span>
      ) : null}
    </li>
  );
}

/**
 * The Planner and the Workers of a resolved stage configuration, Workers in
 * name order. The one presentation of ConsumerExecutionConfig: Workflow
 * pages list what a stage declares, Run pages what an attempt resolved
 * (`exact`).
 */
export function ConsumerConfigList({
  config,
  workerLabel,
  exact = false,
  className,
}: {
  config: Pick<ResolvedConfig, "planner" | "agents">;
  /** The row name of one Worker, e.g. "Agent reviewer". */
  workerLabel: (name: string) => string;
  exact?: boolean;
  className: string;
}) {
  return (
    <ul className={className}>
      {config.planner === undefined ? null : (
        <ConsumerConfigView
          label="Planner"
          config={config.planner}
          exact={exact}
        />
      )}
      {Object.entries(config.agents)
        .sort(([left], [right]) => left.localeCompare(right))
        .map(([name, consumer]) => (
          <ConsumerConfigView
            key={name}
            label={workerLabel(name)}
            config={consumer}
            exact={exact}
          />
        ))}
    </ul>
  );
}
