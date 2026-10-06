import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { describe, expect, it, vi } from "vitest";

import {
  DecisionBar,
  type DecisionBarProps,
  type DecisionOption,
} from "./decision-bar";

const OPTIONS: DecisionOption[] = [
  { id: "confirm", label: "Confirm issue", shortcut: "c", tone: "primary" },
  { id: "reject", label: "Not an issue", shortcut: "r" },
  { id: "evidence", label: "Needs evidence", shortcut: "e" },
];

const SEVERITIES = [
  { value: "low", label: "Low" },
  { value: "high", label: "High" },
];

type HarnessProps = Partial<
  Omit<DecisionBarProps, "rationale" | "severity" | "selected">
> & {
  withSeverity?: boolean;
  initialReason?: string;
  initialSelected?: string;
};

function Harness({
  withSeverity = false,
  initialReason = "",
  initialSelected,
  onSelect,
  ...props
}: HarnessProps) {
  const [selected, setSelected] = useState(initialSelected);
  const [reason, setReason] = useState(initialReason);
  const [severity, setSeverity] = useState<string | undefined>(undefined);
  return (
    <DecisionBar
      options={OPTIONS}
      selected={selected}
      onSelect={(id) => {
        onSelect?.(id);
        setSelected(id);
      }}
      severity={
        withSeverity
          ? {
              options: SEVERITIES,
              value: severity,
              onChange: setSeverity,
              required: selected === "confirm",
              hint: "Pick one when you confirm.",
            }
          : undefined
      }
      rationale={{ value: reason, onChange: setReason, maxLength: 500 }}
      onSubmit={vi.fn()}
      {...props}
    />
  );
}

function recordButton() {
  return screen.getByRole("button", { name: "Record decision" });
}

function reasonField() {
  return screen.getByRole("textbox", { name: "Why" });
}

describe("DecisionBar", () => {
  it("chooses a verdict by its key and focuses the reason", async () => {
    const onSelect = vi.fn();
    render(<Harness onSelect={onSelect} />);
    expect(screen.getByRole("region", { name: "Your decision" })).toBeVisible();
    const reject = screen.getByRole("button", { name: "Not an issue" });
    expect(reject).toHaveAttribute("aria-keyshortcuts", "R");
    expect(reject).toHaveAttribute("aria-pressed", "false");

    await userEvent.setup().keyboard("r");
    expect(onSelect).toHaveBeenCalledWith("reject");
    expect(reject).toHaveAttribute("aria-pressed", "true");
    expect(reasonField()).toHaveFocus();
    expect(reasonField()).toHaveValue("");
  });

  it("chooses a verdict by click and types keys into the reason", async () => {
    render(<Harness />);
    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: "Needs evidence" }));
    expect(reasonField()).toHaveFocus();
    await user.keyboard("crej");
    expect(reasonField()).toHaveValue("crej");
    expect(
      screen.getByRole("button", { name: "Needs evidence" }),
    ).toHaveAttribute("aria-pressed", "true");
  });

  it("records only with a verdict, a reason and a required severity", async () => {
    const onSubmit = vi.fn();
    render(<Harness withSeverity onSubmit={onSubmit} />);
    const user = userEvent.setup();
    expect(recordButton()).toBeDisabled();
    expect(recordButton()).toHaveAccessibleDescription("Choose a decision.");

    await user.keyboard("c");
    expect(recordButton()).toBeDisabled();
    expect(recordButton()).toHaveAccessibleDescription("Choose severity.");
    await user.click(screen.getByRole("radio", { name: "High" }));
    expect(
      screen.getByRole("radiogroup", { name: "Severity" }),
    ).toHaveAttribute("aria-required", "true");
    expect(recordButton()).toHaveAccessibleDescription("Write a short reason.");

    await user.click(reasonField());
    await user.keyboard("   ");
    expect(recordButton()).toBeDisabled();
    await user.keyboard("Scoped by owner");
    expect(recordButton()).toBeEnabled();
    expect(recordButton()).not.toHaveAccessibleDescription();
    await user.click(recordButton());
    expect(onSubmit).toHaveBeenCalledTimes(1);
  });

  it("does not need a severity for verdicts that do not ask for one", async () => {
    render(<Harness withSeverity initialReason="Not reachable" />);
    await userEvent.setup().keyboard("r");
    expect(recordButton()).toBeEnabled();
  });

  it("records with Ctrl+Enter or ⌘+Enter from the reason", async () => {
    const onSubmit = vi.fn();
    render(<Harness onSubmit={onSubmit} />);
    const user = userEvent.setup();
    await user.click(reasonField());
    await user.keyboard("{Control>}{Enter}{/Control}");
    expect(onSubmit).not.toHaveBeenCalled();

    await user.keyboard("e");
    expect(reasonField()).toHaveValue("e");
    await user.click(screen.getByRole("button", { name: "Not an issue" }));
    await user.keyboard("{Control>}{Enter}{/Control}");
    expect(onSubmit).toHaveBeenCalledTimes(1);
    await user.keyboard("{Meta>}{Enter}{/Meta}");
    expect(onSubmit).toHaveBeenCalledTimes(2);
    expect(reasonField()).toHaveValue("e");
    expect(reasonField()).toHaveAttribute(
      "aria-keyshortcuts",
      "Control+Enter Meta+Enter",
    );
  });

  it("disables everything while pending", async () => {
    const onSelect = vi.fn();
    const onSubmit = vi.fn();
    const onNext = vi.fn();
    render(
      <Harness
        pending
        initialSelected="reject"
        initialReason="Duplicate of the login flaw"
        onSelect={onSelect}
        onSubmit={onSubmit}
        next={{ label: "Next item", shortcut: "j", onNext }}
        more={<button type="button">More</button>}
      />,
    );
    const user = userEvent.setup();
    for (const option of OPTIONS) {
      expect(screen.getByRole("button", { name: option.label })).toBeDisabled();
    }
    expect(recordButton()).toBeDisabled();
    expect(recordButton()).toHaveAccessibleDescription("Recording…");
    expect(screen.getByRole("button", { name: "Next item" })).toBeDisabled();
    expect(reasonField()).toHaveAttribute("readonly");
    expect(
      screen.getByRole("region", { name: "Your decision" }),
    ).toHaveAttribute("aria-busy", "true");
    expect(
      screen.getByRole("button", { name: "More" }).closest("[inert]"),
    ).not.toBeNull();

    await user.keyboard("cj");
    await user.click(reasonField());
    await user.keyboard("{Control>}{Enter}{/Control}");
    expect(onSelect).not.toHaveBeenCalled();
    expect(onNext).not.toHaveBeenCalled();
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("shows the error as an alert and the disabled reason as a description", () => {
    const onNext = vi.fn();
    const { rerender } = render(
      <Harness error="The decision was not recorded: conflict." />,
    );
    expect(screen.getByRole("alert")).toHaveTextContent(
      "The decision was not recorded: conflict.",
    );
    rerender(
      <Harness
        disabledReason="Only reviewers can decide."
        next={{ label: "Next item", onNext }}
      />,
    );
    expect(screen.queryByRole("alert")).toBeNull();
    expect(
      screen.getByRole("button", { name: "Confirm issue" }),
    ).toBeDisabled();
    expect(reasonField()).toBeDisabled();
    expect(recordButton()).toHaveAccessibleDescription(
      "Only reviewers can decide.",
    );
    expect(screen.getByRole("button", { name: "Next item" })).toBeEnabled();
  });

  it("moves to the next item by button or key", async () => {
    const onNext = vi.fn();
    render(
      <Harness
        next={{
          label: "Next in inbox: blocked endpoint",
          shortcut: "j",
          onNext,
        }}
      />,
    );
    const user = userEvent.setup();
    const next = screen.getByRole("button", {
      name: "Next in inbox: blocked endpoint",
    });
    expect(next).toHaveAttribute("aria-keyshortcuts", "J");
    await user.click(next);
    await user.keyboard("j");
    expect(onNext).toHaveBeenCalledTimes(2);
  });
});
