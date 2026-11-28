import { useId, useState } from "react";
import type { LLMCost } from "../../api/llm-cost";
import { formatUSD } from "../../app/format";
import { Button, StatusChip } from "../../ui";
import { Icon } from "../../app/icon";
import { RecordedTime } from "../../app/recorded-time";
import { RefreshButton } from "../../app/refresh-button";
import "./cost.css";

const BASIS = {
  reported: "Reported",
  estimated: "Estimate",
  mixed: "Mixed sources",
  none: "No price data",
} as const;

export function CostBasisLabel({ basis }: { basis: LLMCost["basis"] }) {
  return BASIS[basis];
}

export function CostAmount({ amount }: { amount: string | null }) {
  return (
    <span
      title={amount === null ? "Cost unavailable" : `USD ${amount}`}
      className="llm-cost-amount"
    >
      {formatUSD(amount)}
    </span>
  );
}

export function LLMCostCard({
  value,
  onRefresh,
  refreshing = false,
}: {
  value: LLMCost;
  onRefresh?: (() => void) | undefined;
  refreshing?: boolean | undefined;
}) {
  const [open, setOpen] = useState(false),
    [group, setGroup] = useState<"activity" | "models">("activity");
  const detailsId = useId();
  const project = value.scope === "project";
  const coverage = value.coverage;
  const count = (value: string) => BigInt(value).toLocaleString("en-US");
  return (
    <section
      className="llm-cost-card"
      aria-label={`${project ? "Project" : value.scope === "audit" ? "Check" : "Run"} LLM cost`}
    >
      <div className="llm-cost-shell">
        <header className="llm-cost-head">
          <h2>
            <Icon name="coins" size={17} /> LLM cost
          </h2>
          <div className="llm-cost-badges">
            {value.basis !== "none" ? (
              <StatusChip
                tone={value.basis === "reported" ? "info" : "warning"}
                size="sm"
              >
                {BASIS[value.basis]}
              </StatusChip>
            ) : null}
            {coverage !== "complete" ? (
              <StatusChip
                tone={coverage === "partial" ? "warning" : "neutral"}
                size="sm"
              >
                {coverage === "partial"
                  ? "Partial"
                  : coverage === "pending"
                    ? "Updating"
                    : "No cost data"}
              </StatusChip>
            ) : null}
          </div>
        </header>
        <div className="llm-cost-main">
          <div>
            <div
              className="llm-cost-total"
              data-empty={value.amount === null || undefined}
            >
              {value.amount === null ? (
                value.pending ? (
                  "Awaiting cost data"
                ) : (
                  "No recorded cost"
                )
              ) : (
                <CostAmount amount={value.amount} />
              )}
            </div>
            <p className="ui-caption">
              {project
                ? "Project lifetime"
                : value.scope === "audit"
                  ? "All iterations & attempts"
                  : "All Stage attempts"}{" "}
              · USD
            </p>
          </div>
          <Button
            size="sm"
            variant="ghost"
            aria-expanded={open}
            aria-controls={detailsId}
            onClick={() => setOpen(!open)}
          >
            Details{" "}
            <Icon name={open ? "chevron-up" : "chevron-down"} size={14} />
          </Button>
        </div>
        {project &&
        (value.checksAmount !== null || value.standaloneAmount !== null) ? (
          <dl className="llm-cost-split">
            <div>
              <dt>
                Checks <small>Child Runs included</small>
              </dt>
              <dd>
                <CostAmount amount={value.checksAmount} />
              </dd>
            </div>
            <div>
              <dt>Standalone Runs</dt>
              <dd>
                <CostAmount amount={value.standaloneAmount} />
              </dd>
            </div>
          </dl>
        ) : null}
        <p className="llm-cost-coverage ui-caption">
          {value.amount === null
            ? value.historicalGap
              ? "Earlier work has no retained monetary data."
              : value.pending
                ? "Cost will update as reports arrive."
                : "No monetary observations were retained for this work."
            : coverage === "partial"
              ? "Known subtotal · some cost is missing."
              : `${count(value.pricedCalls)} of ${count(value.calls)} recorded calls priced`}
          {value.pending && coverage !== "pending"
            ? " Updating as reports arrive."
            : null}
        </p>
      </div>
      {open ? (
        <div className="llm-cost-details" id={detailsId}>
          <div className="llm-cost-detail-title">
            <h3>Cost breakdown</h3>
            <span className="ui-caption">{BASIS[value.basis]}</span>
            {onRefresh ? (
              <RefreshButton
                size="sm"
                isFetching={refreshing}
                onRefresh={onRefresh}
              />
            ) : null}
          </div>
          <p className="llm-cost-exact">
            Exact known amount{" "}
            <code>
              {value.amount === null ? "Unavailable" : `USD ${value.amount}`}
            </code>
          </p>
          <div
            role="group"
            aria-label="Cost breakdown"
            className="llm-cost-modes"
          >
            <Button
              size="sm"
              variant="ghost"
              aria-pressed={group === "activity"}
              onClick={() => setGroup("activity")}
            >
              By activity
            </Button>
            <Button
              size="sm"
              variant="ghost"
              aria-pressed={group === "models"}
              onClick={() => setGroup("models")}
            >
              By model
            </Button>
          </div>
          {value[group].length ? (
            <ul className="llm-cost-rows">
              {value[group].map((row) => (
                <li key={row.key}>
                  <div>
                    <strong>{row.label}</strong>
                    <small>
                      {count(row.calls)} calls · {BASIS[row.basis]}
                    </small>
                  </div>
                  <div>
                    <CostAmount amount={row.amount} />
                    <code>USD {row.amount}</code>
                  </div>
                </li>
              ))}
            </ul>
          ) : (
            <p className="ui-caption">No priced calls recorded.</p>
          )}
          <div className="llm-cost-source ui-caption">
            <p>
              {value.basis === "reported"
                ? "Provider-reported charges, forwarded by the LLM gateway."
                : value.basis === "estimated"
                  ? "Estimated by the LLM gateway using its configured prices."
                  : value.basis === "mixed"
                    ? "Includes provider-reported charges and gateway estimates."
                    : "No monetary observations have been received."}
            </p>
            {value.historicalGap ? (
              <p>
                Earlier work has no retained monetary data. Current prices are
                not applied to historical tokens.
              </p>
            ) : null}
            {value.unknownReports !== "0" ? (
              <p>
                {count(value.unknownReports)} reports have missing or incomplete
                cost data.
              </p>
            ) : null}
            <p>
              Includes recorded unsuccessful attempts.{" "}
              {value.updatedAt ? (
                <>
                  Last observation <RecordedTime value={value.updatedAt} />.
                </>
              ) : null}
            </p>
          </div>
        </div>
      ) : (
        <div id={detailsId} hidden />
      )}
    </section>
  );
}
