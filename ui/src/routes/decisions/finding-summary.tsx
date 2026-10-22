import { useQuery } from "@tanstack/react-query";
import { useId } from "react";

import { getAuditFinding, type AuditFinding } from "../../api/audits";
import { usePublicAPI } from "../../api/context";
import type { components } from "../../api/generated/public";
import { queryKeys } from "../../api/query-keys";
import { formatBytes, plural } from "../../app/format";
import { Icon } from "../../app/icon";
import { RecordedTime } from "../../app/recorded-time";
import {
  capitalize,
  findingStateLabel,
  severityLabel,
  TERMS,
} from "../../app/vocabulary";
import { Facts, IdChip, MethodChip, StatusChip } from "../../ui";
import { CorrelationMark } from "../issues/correlation-mark";
import { FindingLocations } from "../projects/audits/finding-locations";
import { DecisionMarkdown } from "./markdown";
import {
  firstParagraph,
  httpOperation,
  splitImpact,
  standardReferences,
  weaknessReferences,
} from "./text";

import "./decisions.css";

type ProposalDocument = components["schemas"]["FindingProposalDocument"];
type SemanticAssessment = NonNullable<
  AuditFinding["currentAssessment"]
>["semanticAssessment"];

/** Locations a compact summary lists before "N more". */
const COMPACT_LOCATIONS = 3;

const ASSESSMENT_LABELS: Readonly<Record<SemanticAssessment, string>> = {
  supported: "Supported",
  refuted: "Refuted",
  inconclusive: "Inconclusive",
  blocked: "Blocked",
  satisfied: "Satisfied",
  violated: "Violated",
  "not-tested": "Not tested",
};

interface FindingSummaryProps {
  auditId: string;
  finding: AuditFinding;
  /**
   * "full" (default) for the issue page: every section. "compact" for an
   * Inbox preview: title, facts, the first paragraph, the first three
   * locations and what the AI is unsure about.
   */
  variant?: "full" | "compact" | undefined;
  /** Heading level of the title; sections use the next level. Default "h2". */
  titleAs?: "h2" | "h3" | undefined;
}

function nonBlank(value: string): boolean {
  return value.trim() !== "";
}

/** The original of a duplicate: its title when it can be read, and its ID. */
function DuplicateOriginal({
  auditId,
  findingId,
}: {
  auditId: string;
  findingId: string;
}) {
  const api = usePublicAPI();
  const original = useQuery({
    queryKey: [
      ...queryKeys.audits.detail(auditId),
      "findings",
      findingId,
      "exact",
    ],
    queryFn: () => getAuditFinding(api, auditId, findingId),
    retry: false,
  });
  return (
    <>
      {original.data === undefined ? null : (
        <span className="decisions-fact-text">
          {original.data.firstProposal.document.title}{" "}
        </span>
      )}
      <IdChip value={findingId} label="ID of the original possible issue" />
    </>
  );
}

function FindingFacts({
  auditId,
  finding,
}: {
  auditId: string;
  finding: AuditFinding;
}) {
  const document = finding.firstProposal.document;
  const subject = document.subject;
  const operation = subject === null ? undefined : httpOperation(subject.key);
  const weaknesses = weaknessReferences(document);
  const standards = standardReferences(document);
  const suggestion = document.severity_suggestion;
  return (
    <Facts
      layout="row"
      className="decisions-facts"
      items={[
        {
          term: "Subject",
          value: subject !== null && operation === undefined && (
            <>
              <span className="ui-mono">{subject.key}</span>{" "}
              <span className="ui-caption">{subject.kind}</span>
            </>
          ),
        },
        {
          term: "Endpoint",
          value: operation && (
            <span className="decisions-fact-endpoint">
              <MethodChip method={operation.method} />{" "}
              <span className="ui-mono">{operation.path}</span>
            </span>
          ),
        },
        {
          term: "Weakness",
          value:
            weaknesses.length > 0 &&
            weaknesses.map((reference, index) => (
              <span
                key={`${reference.requirement_id}@${reference.version}`}
                title={`${reference.scheme} ${reference.version}`}
              >
                {index === 0 ? "" : ", "}
                {reference.requirement_id}
              </span>
            )),
        },
        {
          term: standards.length === 1 ? "Standard" : "Standards",
          value: standards.length > 0 && (
            <span className="decisions-fact-list">
              {standards.map((reference) => (
                <span
                  key={`${reference.scheme}@${reference.version}/${reference.requirement_id}`}
                >
                  {reference.requirement_id}{" "}
                  <span className="ui-caption">
                    {reference.scheme}@{reference.version}
                  </span>
                </span>
              ))}
            </span>
          ),
        },
        {
          term: "Severity",
          value: (
            <>
              {severityLabel(finding.analystSeverity)}
              {finding.analystSeverity !== undefined ||
              suggestion === "" ? null : (
                <span className="decisions-suggestion">
                  {" "}
                  · AI suggestion: {severityLabel(suggestion)}
                </span>
              )}
            </>
          ),
        },
        {
          term: "Duplicate of",
          value: finding.duplicateTargetId !== undefined && (
            <DuplicateOriginal
              auditId={auditId}
              findingId={finding.duplicateTargetId}
            />
          ),
        },
        {
          term: "Verification",
          value:
            finding.currentAssessment !== undefined &&
            (ASSESSMENT_LABELS[finding.currentAssessment.semanticAssessment] ??
              capitalize(
                finding.currentAssessment.semanticAssessment.replaceAll(
                  "-",
                  " ",
                ),
              )),
        },
      ]}
    />
  );
}

function Unsure({
  headingId,
  Heading,
  limitations,
  preconditions,
}: {
  headingId: string;
  Heading: "h3" | "h4";
  limitations: readonly string[];
  preconditions: readonly string[];
}) {
  const list = (items: readonly string[]) =>
    items.length === 1 ? (
      <p>{items[0]}</p>
    ) : (
      <ul>
        {items.map((item, index) => (
          <li key={index}>{item}</li>
        ))}
      </ul>
    );
  return (
    <section className="decisions-unsure" aria-labelledby={headingId}>
      <Heading id={headingId} className="decisions-heading">
        <Icon name="question-circle" size={15} />
        What the AI is unsure about
      </Heading>
      {limitations.length === 0 ? null : list(limitations)}
      {preconditions.length === 0 ? null : (
        <>
          <p className="decisions-unsure-label">It depends on:</p>
          {list(preconditions)}
        </>
      )}
    </section>
  );
}

function evidenceLabel(document: ProposalDocument, index: number): string {
  return document.evidence_ids[index] ?? `evidence ${index + 1}`;
}

/**
 * A possible issue as the AI proposed it: state, title and facts, what the AI
 * found, its stated impact, the code and request locations, what it is
 * unsure about, and the retained evidence. Every section shows the
 * proposal's own fields and is left out when they are empty; the analyst's
 * severity is never filled from the AI's suggestion.
 */
export function FindingSummary({
  auditId,
  finding,
  variant = "full",
  titleAs = "h2",
}: FindingSummaryProps) {
  const id = useId();
  const compact = variant === "compact";
  const Title = titleAs;
  const Heading = titleAs === "h2" ? "h3" : "h4";
  const document = finding.firstProposal.document;
  const state = findingStateLabel(finding.state);
  // Possible issues stay apart from confirmed issues (S19:1791-1793).
  const kind =
    finding.state === "confirmed"
      ? capitalize(TERMS.issue)
      : capitalize(TERMS.possibleIssue);
  const { found, impact } = splitImpact(document.description);
  const foundText = compact ? firstParagraph(found) : found;
  const limitations = document.limitations.filter(nonBlank);
  const preconditions = document.preconditions.filter(nonBlank);
  const locations = document.locations ?? [];
  const shown: ProposalDocument = {
    ...document,
    locations: compact ? locations.slice(0, COMPACT_LOCATIONS) : locations,
  };
  if (compact) delete shown.http_exchange;
  const hasLocations =
    (shown.locations?.length ?? 0) > 0 || shown.http_exchange !== undefined;
  const hiddenLocations = locations.length - (shown.locations?.length ?? 0);
  const evidence = compact ? [] : finding.firstProposal.evidence;
  const showImpact = !compact && impact !== undefined;
  const showUnsure = limitations.length > 0 || preconditions.length > 0;

  return (
    <article
      className="decisions-summary"
      data-variant={variant}
      aria-labelledby={`${id}-title`}
    >
      <header className="decisions-summary-head">
        <p className="decisions-summary-kicker">
          <StatusChip tone={state.tone}>{state.label}</StatusChip>
          <CorrelationMark finding={finding} />
          <span>
            <span className="decisions-summary-kind">{kind}</span>
            {" · Found "}
            <RecordedTime value={finding.createdAt} />
          </span>
        </p>
        <Title id={`${id}-title`} className="decisions-summary-title">
          {document.title}
        </Title>
        <FindingFacts auditId={auditId} finding={finding} />
      </header>
      {foundText === "" ? null : (
        <section className="decisions-section" aria-labelledby={`${id}-found`}>
          <Heading id={`${id}-found`} className="decisions-heading">
            What the AI found
          </Heading>
          <DecisionMarkdown source={foundText} />
        </section>
      )}
      {!hasLocations ? null : (
        <div className="decisions-section decisions-locations">
          <Heading className="decisions-heading">Locations</Heading>
          <FindingLocations document={shown} />
          {hiddenLocations <= 0 ? null : (
            <p className="ui-caption">
              {plural(hiddenLocations, "more location")}
            </p>
          )}
        </div>
      )}
      {!showImpact && !showUnsure ? null : (
        <div className="decisions-pair">
          {!showImpact ? null : (
            <section
              className="decisions-section"
              aria-labelledby={`${id}-impact`}
            >
              <Heading id={`${id}-impact`} className="decisions-heading">
                Impact
              </Heading>
              <DecisionMarkdown source={impact} />
            </section>
          )}
          {!showUnsure ? null : (
            <Unsure
              headingId={`${id}-unsure`}
              Heading={Heading}
              limitations={limitations}
              preconditions={preconditions}
            />
          )}
        </div>
      )}
      {evidence.length === 0 ? null : (
        <section
          className="decisions-section"
          aria-labelledby={`${id}-evidence`}
        >
          <Heading id={`${id}-evidence`} className="decisions-heading">
            Evidence
          </Heading>
          <ul className="decisions-rows">
            {evidence.map((artifact, index) => (
              <li
                key={`${artifact.ref.namespace}/${artifact.ref.name}@${artifact.ref.revision}`}
                title={`Revision ${artifact.ref.revision} · ${artifact.digest}`}
              >
                <span className="ui-mono">
                  {evidenceLabel(document, index)}
                </span>
                <span className="ui-mono decisions-row-main">
                  {artifact.ref.namespace}/{artifact.ref.name}
                </span>
                <span className="ui-caption">
                  {artifact.mediaType} · {formatBytes(artifact.sizeBytes)}
                </span>
              </li>
            ))}
          </ul>
        </section>
      )}
    </article>
  );
}
