import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { components } from "../../api/generated/public";
import { CompactionProgress } from "./compaction-progress";

const progress: components["schemas"]["WorkerCompactionProgress"] = {
  invocationId: "worker-1",
  generation: 1,
  phase: "applied",
  attempts: 1,
  failures: 0,
  archiveReads: 0,
  inputBytes: 100,
  summaryBytes: 20,
  reasoningAvailable: 0,
  reasoningIncluded: 0,
  reasoningTruncated: 0,
  beforePrompt: { generation: 0, promptTokens: 600 },
  afterPrompt: { generation: 1 },
};
describe("confirmed compaction observations", () => {
  it("requires applied and an ongoing attempt, and keeps after unavailable until measured", () => {
    const view = render(<CompactionProgress progress={progress} continuing />);
    expect(
      screen.getByText("Контекст сжат, работа продолжается"),
    ).toBeVisible();
    expect(screen.getByText("Поколение 1: нет данных")).toBeVisible();
    view.rerender(
      <CompactionProgress
        progress={{
          ...progress,
          afterPrompt: { generation: 1, promptTokens: 80 },
        }}
      />,
    );
    expect(screen.queryByText("Контекст сжат, работа продолжается")).toBeNull();
    expect(screen.getByText("Поколение 1: 80")).toBeVisible();
    view.rerender(
      <CompactionProgress
        progress={{ ...progress, phase: "failed" }}
        continuing
      />,
    );
    expect(screen.queryByText("Контекст сжат, работа продолжается")).toBeNull();
    expect(screen.getByText("Сжатие завершилось ошибкой")).toBeVisible();
  });
});
