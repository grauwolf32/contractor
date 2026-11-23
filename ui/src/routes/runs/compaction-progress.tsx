import type { components } from "../../api/generated/public";
import { Facts, TechnicalDetails } from "../../ui";

type Progress = components["schemas"]["WorkerCompactionProgress"];
type Phases = components["schemas"]["WorkerModelPhases"];

const PHASE_LABELS = {
  idle: "Сжатие ещё не запрашивалось",
  requested: "Сжатие запрошено",
  running: "Контекст сжимается",
  applied: "Контекст сжат",
  failed: "Сжатие завершилось ошибкой",
};

/** Numeric observations never determine the enclosing attempt's outcome. */
export function CompactionProgress({
  progress,
  phases,
  continuing = false,
}: {
  progress?: Progress | undefined;
  phases?: Phases | undefined;
  continuing?: boolean | undefined;
}) {
  const measurement = (value: Progress["beforePrompt"]) =>
    value === undefined
      ? "Нет данных"
      : `Поколение ${value.generation}: ${value.promptTokens ?? "нет данных"}`;
  return (
    <section aria-label="Сжатие контекста">
      <p>
        {progress?.phase === "applied" && continuing
          ? "Контекст сжат, работа продолжается"
          : progress === undefined
            ? "Расход модели"
            : PHASE_LABELS[progress.phase]}
      </p>
      {progress === undefined ? null : (
        <Facts
          items={[
            { term: "Поколение контекста", value: progress.generation },
            {
              term: "До сжатия, токены запроса",
              value: measurement(progress.beforePrompt),
            },
            {
              term: "После сжатия, токены запроса",
              value: measurement(progress.afterPrompt),
            },
          ]}
        />
      )}
      <TechnicalDetails>
        {progress === undefined ? null : (
          <Facts
            items={[
              {
                term: "Попытки / применено / ошибки",
                value: `${progress.attempts} / ${progress.generation} / ${progress.failures}`,
              },
              { term: "Причина ошибки", value: progress.lastCause },
              {
                term: "Вход / summary, байты",
                value: `${progress.inputBytes} / ${progress.summaryBytes}`,
              },
              { term: "Чтения архива", value: progress.archiveReads },
              {
                term: "Reasoning: доступно / включено / усечено",
                value: `${progress.reasoningAvailable} / ${progress.reasoningIncluded} / ${progress.reasoningTruncated}`,
              },
            ]}
          />
        )}
        {phases === undefined ? <p>Затраты модели недоступны.</p> : null}
        {Object.entries(phases ?? {}).map(([name, usage]) => (
          <div key={name}>
            <strong>{name}</strong>
            <Facts
              items={[
                { term: "Вызовы модели", value: usage.modelCalls },
                {
                  term: "Известные input / output / total токены",
                  value: `${usage.inputTokens} / ${usage.outputTokens} / ${usage.totalTokens}`,
                },
                {
                  term: "Вызовы без usage",
                  value: usage.tokenUsageUnavailable,
                },
                {
                  term: "Время модели, мс",
                  value: usage.durationMs ?? "Нет данных",
                },
              ]}
            />
            {usage.tokenUsageUnavailable > 0 ? (
              <p>Полные затраты токенов недоступны.</p>
            ) : null}
          </div>
        ))}
        <p>
          Стоимость в деньгах — в отчёте затрат Run. Байты не определяют
          экономию токенов.
        </p>
      </TechnicalDetails>
    </section>
  );
}
