import { describe, expect, it } from "vitest";
import shared from "../../../../api/testdata/planner-fact-cases.json";
import { patchDraft, starter } from "./document";
import { PLANNER_FACTS } from "./planners";
import { validateDraft } from "./validation";

describe("shared Planner facts", () => {
  it("lists exactly the registered Planners with their facts", () => {
    expect(PLANNER_FACTS).toEqual(
      Object.fromEntries(
        shared.planners.map(({ ref, singleAgent, agentResults }) => [
          ref,
          { singleAgent, agentResults },
        ]),
      ),
    );
  });

  it.each(shared.cases)(
    "validates $name",
    ({ planner, agents, resultFromAgent, valid }) => {
      const stage = ["spec", "stages", "start"];
      let draft = patchDraft(
        starter("Workflow"),
        [...stage, "planner"],
        planner,
      );
      draft = patchDraft(
        draft,
        [...stage, "agents"],
        Object.fromEntries(
          Array.from({ length: agents }, (_, index) => [
            `worker_${index + 1}`,
            { template: "worker@1", namespace: `work_${index + 1}` },
          ]),
        ),
      );
      draft = patchDraft(draft, [...stage, "result", "artifacts"], {
        report: {
          required: true,
          mediaTypes: ["application/json"],
          from: {
            namespace: resultFromAgent ? "work_1" : "elsewhere",
            name: "report.json",
          },
        },
      });
      const errors = validateDraft(draft).filter(
        (problem) => problem.severity === "error",
      );
      if (valid) expect(errors).toEqual([]);
      else expect(errors).not.toEqual([]);
    },
  );
});
