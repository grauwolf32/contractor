import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { PublicAPIError } from "../api/error";
import { writeConflict } from "./conflict";
import { ConflictNotice } from "./conflict-notice";

function refused(status: number): PublicAPIError {
  return new PublicAPIError({ status, code: "refused", message: "Refused" });
}

describe("write conflicts", () => {
  it("tells a stale revision (412) from a conflicting state (409)", () => {
    expect(writeConflict(refused(412))).toBe("stale");
    expect(writeConflict(refused(409))).toBe("conflict");
    expect(writeConflict(refused(404))).toBeUndefined();
    expect(writeConflict(refused(503))).toBeUndefined();
    expect(writeConflict(new Error("offline"))).toBeUndefined();
    expect(writeConflict(undefined)).toBeUndefined();
  });

  it("says what happened in the user's words and nothing for other errors", () => {
    const { rerender, container } = render(
      <ConflictNotice error={refused(412)} subject="The check" />,
    );
    expect(screen.getByRole("status")).toHaveTextContent(
      "The check changed in the meantime. The page now shows its current state; review it before trying again.",
    );
    rerender(<ConflictNotice error={refused(409)} subject="This credential" />);
    expect(screen.getByRole("status")).toHaveTextContent(
      "This credential is no longer in a state that allows this, so nothing was applied.",
    );
    rerender(<ConflictNotice error={refused(500)} subject="The check" />);
    expect(container).toBeEmptyDOMElement();
  });
});
