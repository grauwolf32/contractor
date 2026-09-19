import { render, renderHook, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import type { components } from "../../../api/generated/public";
import type { PerformanceHistory } from "../../../api/performance";
import { GPUCurrentMetrics, GPUHistoryCharts } from "./gpu";
import { useGPUColors } from "./gpu-colors";

const colors = new Map([
  ["GPU-aa", "#d8ff72"],
  ["GPU-bb", "#77c7ff"],
]);

const at = "2026-09-19T12:00:00Z";
const freshness: components["schemas"]["PerformanceFreshness"] = {
  status: "ok",
  observedAt: at,
  lastAttemptAt: at,
  intervalSeconds: 15,
  coverage: {
    startedAt: "2026-09-19T11:59:45Z",
    endedAt: at,
    durationSeconds: 15,
    expectedSamples: 1,
    observedSamples: 1,
  },
};
const gpu: components["schemas"]["PerformanceGPU"] = {
  freshness,
  devices: [
    {
      id: "GPU-aa",
      name: "NVIDIA GPU A",
      utilizationPercent: 0,
      memoryUsedBytes: 1024 ** 3,
      memoryTotalBytes: 32 * 1024 ** 3,
    },
  ],
};

describe("optional GPU observations", () => {
  it("retains distinct UUID colors through reorder, missing readings and range changes", () => {
    const ids = Array.from({ length: 8 }, (_, i) => `GPU-${i}`);
    const { result, rerender } = renderHook(({ ids }) => useGPUColors(ids), {
      initialProps: { ids },
    });
    const original = result.current;
    expect(new Set(ids.map((id) => original.get(id))).size).toBe(8);
    rerender({ ids: [...ids].reverse() });
    expect(result.current).toBe(original);
    rerender({ ids: ["GPU-7"] });
    expect(result.current.get("GPU-7")).toBe(original.get("GPU-7"));
    rerender({ ids: ["GPU-ff", ...ids] });
    for (const id of ids) expect(result.current.get(id)).toBe(original.get(id));
    expect([...original.values()]).not.toContain(result.current.get("GPU-ff"));
  });

  it("keeps returning and replacement GPUs distinct after palette reuse", () => {
    const { result, rerender } = renderHook(({ ids }) => useGPUColors(ids), {
      initialProps: { ids: [] as string[] },
    });
    for (let batch = 0; batch < 5; batch++) {
      const ids = Array.from(
        { length: 8 },
        (_, i) => `GPU-${(batch * 8 + i).toString(16)}`,
      );
      rerender({ ids });
      expect(new Set(ids.map((id) => result.current.get(id))).size).toBe(8);
      expect(result.current.size).toBeLessThanOrEqual(16);
    }
    const returning = ["GPU-0", "GPU-1", "GPU-22", "GPU-23"];
    rerender({ ids: returning });
    expect(new Set(returning.map((id) => result.current.get(id))).size).toBe(4);
  });

  it("hides absent, unavailable and wholly unsupported GPU measurements", () => {
    const { container, rerender } = render(
      <GPUCurrentMetrics colors={colors} gpu={undefined} readAt={at} />,
    );
    expect(container).toBeEmptyDOMElement();
    rerender(
      <GPUCurrentMetrics
        colors={colors}
        gpu={{
          freshness: {
            status: "unavailable",
            reason: "read_failed",
            lastAttemptAt: at,
            intervalSeconds: 15,
            coverage: { ...freshness.coverage, observedSamples: 0 },
          },
          devices: [],
        }}
        readAt={at}
      />,
    );
    expect(container).toBeEmptyDOMElement();
    rerender(
      <GPUCurrentMetrics
        colors={colors}
        gpu={{
          freshness: {
            ...freshness,
            status: "partial",
            reason: "unsupported_metric",
          },
          devices: [{ id: "GPU-aa", name: "NVIDIA GPU A" }],
        }}
        readAt={at}
      />,
    );
    expect(container).toBeEmptyDOMElement();
  });

  it("shows real zero and supported fields, omitting unsupported sensors", () => {
    render(<GPUCurrentMetrics colors={colors} gpu={gpu} readAt={at} />);
    expect(
      screen.getByRole("article", { name: "GPU NVIDIA GPU A · aa" }),
    ).toBeVisible();
    expect(screen.getByText("0 %")).toBeVisible();
    expect(screen.getByText("VRAM used")).toBeVisible();
    expect(screen.getByText("VRAM capacity")).toBeVisible();
    expect(screen.queryByText("Temperature")).not.toBeInTheDocument();
    expect(screen.queryByText("Power draw")).not.toBeInTheDocument();
    expect(screen.queryByText("Power limit")).not.toBeInTheDocument();
    expect(screen.getByText(/includes other applications/)).toBeVisible();
  });

  it("does not present an old reading as current", () => {
    render(
      <GPUCurrentMetrics
        colors={colors}
        gpu={gpu}
        readAt="2026-09-19T12:02:00Z"
      />,
    );
    expect(screen.getByText("stale")).toBeVisible();
  });

  it("hides empty GPU history and preserves gaps, identities and restarts", () => {
    const history: PerformanceHistory = {
      from: "2026-09-19T11:59:00Z",
      to: "2026-09-19T12:01:00Z",
      step: "15s",
      points: [],
    };
    const { container, rerender } = render(
      <GPUHistoryCharts colors={colors} history={history} stepSeconds={15} />,
    );
    expect(container).toBeEmptyDOMElement();
    history.points = [0, 1, 2, 3].map((i) => ({
      kind: "sample",
      version: 1,
      generation: i === 3 ? "after" : "before",
      observedAt: new Date(Date.parse(at) - (3 - i) * 15_000).toISOString(),
      ...(i === 1
        ? {}
        : {
            gpu: {
              ...gpu,
              devices:
                i === 2
                  ? [
                      {
                        id: "GPU-bb",
                        name: "NVIDIA GPU B",
                        utilizationPercent: 90,
                      },
                      ...gpu.devices,
                    ]
                  : gpu.devices,
            },
          }),
    }));
    rerender(
      <GPUHistoryCharts colors={colors} history={history} stepSeconds={15} />,
    );
    const first = screen.getByRole("img", {
      name: "GPU utilization",
    });
    const lines = first.querySelectorAll<SVGPolylineElement>("polyline");
    expect(new Set([...lines].map((line) => line.style.stroke)).size).toBe(2);
    expect(
      screen.getAllByRole("img", { name: "GPU utilization" }),
    ).toHaveLength(1);
    expect(
      screen.getByLabelText("GPU utilization GPU legend"),
    ).toHaveTextContent("NVIDIA GPU A · aa0 %NVIDIA GPU B · bb90 %");
    expect(
      first.querySelectorAll('[data-series-id="GPU-aa"] polyline'),
    ).toHaveLength(3);
    expect(
      screen.getByLabelText(
        "GPU utilization · NVIDIA GPU A · aa numeric summary",
      ),
    ).toHaveTextContent("Observed points3");
    expect(
      screen.getByLabelText(
        "GPU utilization · NVIDIA GPU B · bb numeric summary",
      ),
    ).toHaveTextContent("Latest90 %");
    expect(
      screen.queryByRole("img", { name: /Temperature|Power draw/ }),
    ).not.toBeInTheDocument();
    expect(
      screen.getByLabelText("VRAM used · NVIDIA GPU A · aa numeric summary"),
    ).toHaveTextContent("Latest1,024 MiB");
  });

  it("renders the last observed aggregate value without inventing unsupported metrics", () => {
    const gauge = { min: 0, max: 90, last: 40, samples: 3, observedAt: at };
    const history: PerformanceHistory = {
      from: "2026-09-19T11:00:00Z",
      to: "2026-09-19T12:01:00Z",
      step: "5m",
      points: [
        {
          kind: "aggregate",
          version: 1,
          generation: "a",
          minuteStart: "2026-09-19T12:00:00Z",
          status: "partial",
          coverageSeconds: 45,
          omittedWindows: 0,
          droppedMinutes: 0,
          stepSeconds: 300,
          observedMinutes: 1,
          expectedMinutes: 5,
          process: {},
          pool: {},
          gpu: {
            freshness,
            devices: [
              { id: "GPU-aa", name: "NVIDIA GPU A", utilizationPercent: gauge },
            ],
          },
        },
      ],
    };
    render(
      <GPUHistoryCharts colors={colors} history={history} stepSeconds={300} />,
    );
    expect(
      screen.getByLabelText(
        "GPU utilization · NVIDIA GPU A · aa numeric summary",
      ),
    ).toHaveTextContent("Latest40 %");
    expect(screen.getByRole("img")).toHaveAccessibleDescription(
      /last observation in each interval/,
    );
  });
});
