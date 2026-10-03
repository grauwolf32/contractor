import eventSchema from "../../../api/events/contractor-events-v1.schema.json";
import { describe, expect, it } from "vitest";

import { parseServerFrame } from "./run-events";

const SERVER_FRAME_TYPES = new Set([
  "subscribed",
  "unsubscribed",
  "event",
  "resync_required",
  "error",
]);

// The Go server validates these schema examples; the UI parser must accept
// every server-sent one.
const serverExamples = eventSchema.examples.filter((example) =>
  SERVER_FRAME_TYPES.has(example.type),
);

describe("event schema examples", () => {
  it("include server frames", () => {
    expect(serverExamples.length).toBeGreaterThan(0);
  });

  it.each(serverExamples.map((example) => [example.type, example] as const))(
    "parses the %s example",
    (_type, example) => {
      expect(parseServerFrame(JSON.stringify(example))).toMatchObject({
        type: example.type,
        subscriptionId: example.subscriptionId,
      });
    },
  );
});
