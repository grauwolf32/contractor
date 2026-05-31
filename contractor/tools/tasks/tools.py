from __future__ import annotations

import json
import logging
from contextlib import suppress
from typing import Any, Callable, Final, Literal, Optional, Union

from google.adk.agents import LlmAgent
from google.adk.tools import AgentTool
from google.adk.tools.tool_context import ToolContext
from pydantic import ValidationError

from contractor.tools.tasks.formatters import (
    SubtaskFormatter,
    _is_empty_worker_response,
    _parse_worker_output,
    _stringify_formatted,
)
from contractor.tools.tasks.manager import StreamlineManager
from contractor.tools.tasks.models import (
    DO_NOT_FINISH_WITH_NO_TASKS_DONE,
    NO_ACTIVE_SUBTASKS_MSG,
    NO_REMAINING_SUBTASKS_MSG,
    NO_SUBTASKS_EXIST_MSG,
    SKIP_REASON_MUST_NOT_BE_EMPTY,
    SUBTASK_DECOMPOSE_EMPTY_LIST,
    SUBTASK_DECOMPOSE_NOT_DECOMPOSABLE,
    SUBTASK_NOT_CURRENT_MSG,
    SUBTASK_REQUIRES_DECOMPOSITION_MSG,
    SUBTASK_REQUIRES_RESOLUTION_MSG,
    SUBTASK_RESULT_MALFORMED,
    TASK_LIMIT_REACHED_MSG,
    Subtask,
    SubtaskDecomposition,
    SubtaskExecutionResult,
    SubtaskSpec,
    _SUBTASK_DECOMPOSITION_SCHEMA_JSON,
)

logger = logging.getLogger(__name__)

# Sentinel attribute used by `instrument_worker` to snapshot the original
# instruction text and stay idempotent across repeated calls.
_INSTRUCTION_SNAPSHOT_ATTR: Final[str] = "_streamline_original_instruction"


# ═══════════════════════════════════════════════════════════════════
# Instructions & instrumentation
# ═══════════════════════════════════════════════════════════════════
TASK_RESULT_SUMMARIZATION_INSTRUCTIONS: Final[str] = """\
You are a precise technical summarizer. Your job is to produce a clear, \
factual summary of a completed task.

INPUT:
You will receive a JSON object with:
- "objective": The original goal of the task
- "records": A list of subtask execution records
- "result": The final reported result
- "status": The final task status ("done" or "failed")

OUTPUT REQUIREMENTS:
Write a single, structured summary covering:
1. **Objective**: What was the task trying to achieve?
2. **Approach**: What major steps were taken? (bullet list)
3. **Outcome**: What was the final result? Was the objective met?
4. **Status**: Final status and justification
5. **Notable issues** (only if applicable): Blockers, warnings, partial \
failures, or important caveats

RULES:
- Be factual. Do NOT speculate or add information not present in the input.
- Use past tense for completed actions.
- If status is "failed", clearly explain WHAT failed and WHY.
""".strip()


_WORKER_EXAMPLE_DONE: Final[SubtaskExecutionResult] = SubtaskExecutionResult(
    task_id="1",
    status="done",
    output=(
        "- Inspected the requested artifacts:\n"
        "  - artifacts/source_a\n"
        "  - artifacts/source_b\n"
        "- Extracted entries:\n"
        "  - entry_1: value_a\n"
        "  - entry_2: value_b\n"
        "  - entry_3: value_c\n"
        "- Verified the extracted entries against the source artifacts."
    ),
    summary=(
        "Goal: Extract the requested entries from the listed artifacts. "
        "Result: Found 3 verified entries across 2 artifacts."
    ),
)
_WORKER_EXAMPLE_INCOMPLETE: Final[SubtaskExecutionResult] = SubtaskExecutionResult(
    task_id="2",
    status="incomplete",
    output=(
        "- Inspected artifacts/source_a and extracted 1 entry:\n"
        "  - entry_1: value_a\n"
        "- Did not yet inspect: artifacts/source_b, artifacts/source_c, "
        "and 2 other artifacts referenced by the subtask."
    ),
    summary=(
        "Goal: Extract the requested entries from the listed artifacts. "
        "Status: Incomplete — only 1 of 5 artifacts inspected."
    ),
)


def _prepare_worker_instructions(fmt: SubtaskFormatter, type_hint: bool = False) -> str:
    format_description = _stringify_formatted(fmt.format_subtask_result_description())
    ex_done_fmt = _stringify_formatted(
        fmt.format_subtask_result(_WORKER_EXAMPLE_DONE, type_hint=type_hint)
    )
    ex_incomplete_fmt = _stringify_formatted(
        fmt.format_subtask_result(_WORKER_EXAMPLE_INCOMPLETE, type_hint=type_hint)
    )

    return f"""\
CORE RULE:
Finish the requested subtask. Avoid premature termination.
If information is missing, try to obtain or infer it using available tools.
Only stop when the final deliverable is produced or you are genuinely blocked.

STATUS RULES:
- task_id: Copy the exact task_id from the input.
- status:
  - Use 'done' ONLY if the requested deliverable is fully produced and no obvious requested work remains.
  - Make your best effort to complete the assigned task; only return 'incomplete' when work genuinely remains and decomposition is needed.
- output: Include only concrete results from work actually performed (not plans or intentions).
- summary: State the goal, what was completed, and, if incomplete, exactly what remains and why.

OUTPUT RULES:
- The output must contain findings, results, or observations — not a plan.
- Provide concrete evidence from work performed.
- Be specific about what was examined, what was found, and what was not found.
- If incomplete, explicitly state:
  - what has been completed so far
  - what remains unresolved
  - why it remains unresolved (blocking reason)
- Do NOT invent facts, findings, paths, entities, or results.
- If something could not be verified, say so explicitly.
- Use only information supported by the work you actually performed.
- Return ONLY the structured result.

RESPONSE FORMAT (MANDATORY):
After completing the subtask, you MUST return your result using EXACTLY \
the following structure. Do NOT add any text before or after.

FIELD DESCRIPTIONS:
{format_description}

EXAMPLE — Completed subtask:
{ex_done_fmt}

EXAMPLE — Incomplete subtask:
{ex_incomplete_fmt}
"""


def _get_agent_ref(worker: Union[LlmAgent, AgentTool]) -> LlmAgent:
    """Extract the underlying LlmAgent from an AgentTool or return as-is."""
    if isinstance(worker, AgentTool):
        inner = worker.agent
        if not isinstance(inner, LlmAgent):
            raise TypeError(
                f"AgentTool wraps non-LlmAgent ({type(inner).__name__}); "
                f"instrument_worker and task_tools require an LlmAgent worker."
            )
        return inner
    return worker


def instrument_worker(
    worker: LlmAgent,
    fmt: SubtaskFormatter,
    type_hint: bool = False,
    use_input_schema: bool = True,
    use_output_schema: bool = True,
) -> AgentTool:
    """Attach Subtask schemas and worker instructions, then wrap as AgentTool.

    Idempotent: the original `instruction` is snapshotted on first call so
    repeated invocations replace (rather than concatenate) the appended
    instructions.
    """
    if use_input_schema or fmt.format == "json":
        worker.input_schema = Subtask
    if use_output_schema:
        worker.output_schema = SubtaskExecutionResult

    if not hasattr(worker, _INSTRUCTION_SNAPSHOT_ATTR):
        setattr(worker, _INSTRUCTION_SNAPSHOT_ATTR, worker.instruction)
    base_instruction = getattr(worker, _INSTRUCTION_SNAPSHOT_ATTR)
    worker.instruction = base_instruction + _prepare_worker_instructions(
        fmt, type_hint=type_hint
    )
    return AgentTool(worker)


# ═══════════════════════════════════════════════════════════════════
# Tool factory
# ═══════════════════════════════════════════════════════════════════
def task_tools(
    name: str,
    max_tasks: int,
    worker: Union[LlmAgent, AgentTool],
    fmt: SubtaskFormatter,
    *,
    use_skip: bool = True,
    use_type_hint: bool = False,
    use_input_schema: bool = True,
    use_output_schema: bool = True,
    use_summarization: bool = True,
    worker_instrumentation: bool = True,
    max_records: int = 20,
    n_retries: int = 3,
) -> list[Callable[..., Any]]:
    if worker_instrumentation:
        agent_ref = _get_agent_ref(worker)
        worker = instrument_worker(
            agent_ref, fmt, use_type_hint, use_input_schema, use_output_schema
        )

    if not isinstance(worker, AgentTool):
        worker = AgentTool(worker)

    mgr = StreamlineManager(name, max_tasks, fmt)

    # Pre-create summarizer if needed
    summarizer_tool: Optional[AgentTool] = None
    if use_summarization:
        agent_ref = _get_agent_ref(worker)
        summarizer_agent = LlmAgent(
            name="task_summarizer",
            description="Produces structured summaries of completed task executions.",
            instruction=TASK_RESULT_SUMMARIZATION_INSTRUCTIONS,
            tools=agent_ref.tools,
            model=agent_ref.model,
        )
        summarizer_tool = AgentTool(summarizer_agent)

    # ── Tool functions ──────────────────────────────────────────────

    def add_subtask(
        title: str, description: str, tool_context: ToolContext
    ) -> dict[str, Any]:
        """Add a new subtask to the execution plan.

        Creates a subtask with the given title and description. The subtask
        will be executed when it becomes current (all preceding subtasks
        are done or skipped).

        Args:
            title: Concise, action-oriented title in imperative mood.
            description: Detailed description including what work to do,
                available inputs, expected output, and constraints.

        Returns:
            The created subtask on success, or an error if the task limit
            has been reached.

        Before calling:
        - Review `get_records` to confirm work is not already done
        - Confirm the subtask produces NEW information
        """
        subtask: Optional[Subtask] = mgr.add_subtask(
            SubtaskSpec(title=title, description=description), tool_context
        )
        if subtask is None:
            return {"error": TASK_LIMIT_REACHED_MSG.format(max_tasks=max_tasks)}
        return {"result": fmt.format_subtask(subtask, type_hint=use_type_hint)}

    def get_current_subtask(tool_context: ToolContext) -> dict[str, Any]:
        """Get the current subtask to execute next.

        Returns:
            The current subtask, or an error if none exist.

        You MUST call this before `execute_current_subtask`.
        """
        subtask: Optional[Subtask] = mgr.get_current_subtask(tool_context)
        if subtask is None:
            return {"error": NO_SUBTASKS_EXIST_MSG}
        return {"result": fmt.format_subtask(subtask, type_hint=use_type_hint)}

    def list_subtasks(
        tool_context: ToolContext,
        view: Literal["remaining", "all"] = "remaining",
    ) -> dict[str, Any]:
        """Inspect the execution plan without taking action.

        DEFAULT:
        - `view="remaining"` returns only the remaining planned subtasks:
        the current subtask and any later subtasks.

        OPTIONAL:
        - `view="all"` returns the full subtask history, including resolved, decomposed and
        historical subtasks.

        Args:
            view:
                - "remaining": current and future subtasks only
                - "all": full ordered history

        Returns:
            Ordered list of visible subtasks, or an explicit no-remaining-work
            message when the remaining plan is empty.
        """
        if view == "all":
            visible_subtasks = mgr.get_subtasks(tool_context)
        else:
            visible_subtasks = mgr.get_remaining_subtasks(tool_context)
            if not visible_subtasks:
                return {"result": NO_REMAINING_SUBTASKS_MSG}

        return {
            "result": fmt.format_subtasks(
                visible_subtasks,
                type_hint=use_type_hint,
            )
        }

    def get_records(tool_context: ToolContext) -> dict[str, Any]:
        """Retrieve execution records from completed subtasks.

        Returns:
            List of task records (most recent last), capped at max_records.
        """
        records = mgr.get_records(tool_context)[-max_records:]
        return {"result": records}

    def decompose_subtask(
        task_id: str,
        decomposition: SubtaskDecomposition,
        tool_context: ToolContext,
    ) -> dict[str, Any]:
        """Break the current subtask into 1-3 smaller executable subtasks.

        Use only when the current subtask has status 'incomplete' or 'malformed'
        and multiple distinct steps remain.

        Args:
            task_id: MUST match the current subtask exactly.
            decomposition: Ordered list of 1-3 subtasks covering all
                remaining work.
        """
        # Models often emit the decomposition as a JSON string instead of a
        # structured arg; parse it rather than bouncing a wasted turn back.
        if isinstance(decomposition, str):
            try:
                decomposition = json.loads(decomposition)
            except (ValueError, TypeError):
                return {
                    "error": (
                        "TypeError: 'decomposition' must be a SubtaskDecomposition "
                        "object, not a string. Expected schema:\n"
                        f"{_SUBTASK_DECOMPOSITION_SCHEMA_JSON}"
                    )
                }
        # Models often emit the bare subtask array instead of the wrapper
        # object; accept that shape rather than crashing on `.subtasks`.
        if isinstance(decomposition, list):
            decomposition = {"subtasks": decomposition}
        if isinstance(decomposition, dict):
            try:
                decomposition = SubtaskDecomposition.model_validate(decomposition)
            except ValidationError as exc:
                return {"error": f"Validation error in decomposition: {exc}"}
        if not isinstance(decomposition, SubtaskDecomposition):
            return {
                "error": (
                    "TypeError: 'decomposition' must be a SubtaskDecomposition "
                    f"object. Expected schema:\n{_SUBTASK_DECOMPOSITION_SCHEMA_JSON}"
                )
            }

        current = mgr.get_current_subtask(tool_context)
        if current is None:
            return {"error": NO_SUBTASKS_EXIST_MSG}
        if str(task_id) != current.task_id:
            return {"error": SUBTASK_NOT_CURRENT_MSG.format(task_id=task_id)}
        if current.status not in ("incomplete", "malformed"):
            return {
                "error": SUBTASK_DECOMPOSE_NOT_DECOMPOSABLE.format(
                    task_id=task_id, status=current.status
                )
            }

        insertion: Optional[list[Subtask]] = mgr.decompose_current_subtask(
            decomposition.subtasks, tool_context
        )
        if insertion is None:
            return {"error": TASK_LIMIT_REACHED_MSG.format(max_tasks=max_tasks)}
        if len(insertion) == 0:
            return {"error": SUBTASK_DECOMPOSE_EMPTY_LIST}
        return {"result": fmt.format_subtasks(insertion)}

    def skip(task_id: str, reason: str, tool_context: ToolContext) -> dict[str, Any]:
        """Skip execution of the current subtask.

        Marks the current subtask as 'skipped' and advances to the next one.
        If all objectives are achived, use `finish` tool instead.

        Args:
            task_id: Must match the current subtask's task_id exactly.
            reason: Clear, specific explanation of why this subtask is being
                    skipped. Generic reasons like "not needed" or "too hard"
                    are NOT acceptable.

        Returns:
            The next subtask, or a message if no more remain.

        IMPORTANT CONSTRAINTS:
            - You MUST have attempted execution first or have clear evidence
              the subtask cannot produce useful results.
            - Valid reasons: duplicate of another subtask, dependency
              unavailable, provably out of scope, already covered by
              another subtask's output.
            - INVALID reasons: "difficult", "not sure", "might not work",
              "seems unnecessary".
            - For 'malformed' subtasks: prefer `decompose_subtask` over skip,
              since malformed output may contain useful partial information.
            - For 'incomplete' subtasks: you MUST decompose unless this is the
              last remaining subtask, in which case skip is allowed.
        """
        if not reason.strip():
            return {"error": SKIP_REASON_MUST_NOT_BE_EMPTY}

        current = mgr.get_current_subtask(tool_context)
        if current is None:
            return {"result": NO_ACTIVE_SUBTASKS_MSG}
        if str(task_id) != current.task_id:
            return {"error": SUBTASK_NOT_CURRENT_MSG.format(task_id=task_id)}

        subtasks = mgr.get_subtasks(tool_context)
        is_last_subtask = len(subtasks) > 0 and subtasks[-1].task_id == current.task_id
        limit_reached = len(subtasks) >= mgr.max_tasks

        if current.status == "incomplete" and not is_last_subtask and not limit_reached:
            return {
                "error": (
                    f"Subtask `{task_id}` has status 'incomplete' and cannot be "
                    f"skipped unless it is the last remaining subtask. Call `decompose_subtask` on {task_id}."
                )
            }

        next_subtask = mgr.skip(reason, tool_context)
        if next_subtask is None:
            return {"result": NO_ACTIVE_SUBTASKS_MSG}
        return {"result": "ok", "next-subtask": fmt.format_subtask(next_subtask)}

    async def execute_current_subtask(
        tool_context: ToolContext,
    ) -> dict[str, Any]:
        """Execute the current subtask using the worker agent.

        Prerequisites:
            - At least one subtask must exist
            - Current subtask must have status 'new'
            - If 'incomplete'/'malformed', resolve it by decomposing or skipping first

        Returns:
            Record of execution with optional action guidance.

        After this tool returns:
            - If status is 'done': The next subtask becomes current automatically when one exists.
            - If status is 'incomplete': You MUST call `decompose_subtask` or `skip` before proceeding.
            - If status is 'malformed': The raw output has been stored, but the
            result could not be fully parsed. You MUST call `decompose_subtask`
            or `skip` before proceeding.
        """
        current = mgr.get_current_subtask(tool_context)
        if current is None:
            return {"error": NO_ACTIVE_SUBTASKS_MSG}

        match current.status:
            case "malformed":
                return {
                    "error": SUBTASK_REQUIRES_RESOLUTION_MSG.format(
                        task_id=current.task_id,
                        status=current.status,
                    )
                }
            case "incomplete":
                return {
                    "error": SUBTASK_REQUIRES_DECOMPOSITION_MSG.format(
                        task_id=current.task_id,
                    )
                }
            case "done" | "skipped" | "decomposed":
                return {"error": NO_ACTIVE_SUBTASKS_MSG}
            case "new":
                pass
            case _:
                logger.warning(
                    "Unknown subtask status",
                    extra={
                        "task_id": current.task_id,
                        "status": current.status,
                    },
                )
                return {
                    "error": (
                        f"Subtask `{current.task_id}` has unsupported "
                        f"status '{current.status}'."
                    )
                }

        logger.info(
            "Executing subtask",
            extra={"task_id": current.task_id, "title": current.title},
        )

        # Prepare worker input
        if fmt.format == "json" or use_input_schema:
            args: dict[str, Any] = fmt._subtask_to_json(current)
        else:
            args = {"request": fmt.format_subtask(current)}

        # Run worker with retries on empty, unparseable, or task_id-mismatched
        # responses. `n_retries` is the total attempt budget (not extra tries
        # on top of the first call).
        raw: Any = None
        subtask_result: Optional[SubtaskExecutionResult] = None
        malformed_reason: Optional[str] = None

        for attempt in range(1, n_retries + 1):
            raw = await worker.run_async(args=args, tool_context=tool_context)

            if _is_empty_worker_response(raw):
                logger.debug(
                    "Worker returned empty response, retrying",
                    extra={
                        "task_id": current.task_id,
                        "attempt": attempt,
                        "max": n_retries,
                    },
                )
                continue

            candidate = _parse_worker_output(raw, fmt, current.task_id)
            if candidate is None:
                logger.debug(
                    "Worker returned unparseable response, retrying",
                    extra={
                        "task_id": current.task_id,
                        "attempt": attempt,
                        "max": n_retries,
                    },
                )
                malformed_reason = None
                continue

            if candidate.task_id != current.task_id:
                logger.warning(
                    "Worker returned mismatched task_id",
                    extra={
                        "expected": current.task_id,
                        "got": candidate.task_id,
                        "attempt": attempt,
                    },
                )
                malformed_reason = (
                    f"Worker returned result for task_id='{candidate.task_id}' "
                    f"but expected '{current.task_id}'.\n\n"
                    f"Original parsed output:\n{candidate.output}"
                )
                continue

            subtask_result = candidate
            break
        else:
            logger.warning(
                "Worker exhausted retries without a valid result",
                extra={"task_id": current.task_id, "attempts": n_retries},
            )

        # ── Apply malformed fallback ─────────────────────────────────
        raw_dump: Any = raw
        if subtask_result is None:
            if malformed_reason is not None:
                raw_dump = malformed_reason
            with suppress(ValueError, TypeError):
                raw_dump = json.dumps(raw_dump, ensure_ascii=False)

            logger.warning(
                "Failed to parse worker output",
                extra={
                    "task_id": current.task_id,
                    "raw_type": type(raw).__name__,
                },
            )
            runtime_result = {
                "task_id": current.task_id,
                "status": "malformed",
                "output": str(raw_dump),
                "summary": SUBTASK_RESULT_MALFORMED,
            }
            success, error_msg = mgr.complete_current_subtask_from_runtime_result(
                runtime_result, tool_context
            )
            record: dict[str, Any] = {
                **current.model_dump(),
                **runtime_result,
            }
            response: dict[str, Any] = {
                "record": record,
                "error": SUBTASK_RESULT_MALFORMED,
                "action": SUBTASK_REQUIRES_RESOLUTION_MSG.format(
                    task_id=current.task_id,
                    status="malformed",
                ),
            }
            if not success and error_msg:
                response["error"] = error_msg
            return response

        # ── Apply validated result ───────────────────────────────────
        assert subtask_result is not None  # narrows type after the None-branch return above
        success, error_msg = mgr.complete_current_subtask(subtask_result, tool_context)

        record = fmt.format_task_record(current, subtask_result)
        response: dict[str, Any] = {"record": record}

        if not success and error_msg:
            response["error"] = error_msg

        # Inspect the post-update state directly instead of inferring it from idx math.
        next_current = mgr.get_current_subtask(tool_context)
        has_active_subtask = next_current is not None and next_current.status == "new"

        action_parts: list[str] = []
        if subtask_result.status == "incomplete":
            action_parts.append(
                SUBTASK_REQUIRES_DECOMPOSITION_MSG.format(
                    task_id=current.task_id,
                )
            )
        elif not has_active_subtask:
            action_parts.append(NO_ACTIVE_SUBTASKS_MSG)

        if action_parts:
            response["action"] = " ".join(action_parts)

        return response

    async def finish(
        status: Literal["done", "failed"],
        result: str,
        tool_context: ToolContext,
    ) -> dict[str, Any]:
        """Finalize the overall task and report the final outcome.

        Args:
            status: "done" if objective fully achieved, "failed" otherwise.
            result: Comprehensive, self-contained description of outcome.
                    MUST include:
                    - What was accomplished (specific deliverables, data, changes)
                    - What was NOT accomplished (if status is "failed")
                    - All information required by the original task description
                    - Follows the OUTPUT FORMAT specified in the task, if any
                    This result must be understandable WITHOUT access to
                    intermediate notes or execution records.

        When to use:
            - All subtasks done/skipped and goal achieved → "done"
            - Critical blocker prevents completion → "failed"
        """

        subtasks = mgr.get_subtasks(tool_context)
        if status == "done":
            has_new = any(t.status == "new" for t in subtasks)
            has_any = len(subtasks) > 0
            if has_new or not has_any:
                return {"error": DO_NOT_FINISH_WITH_NO_TASKS_DONE}

        summary = ""
        if use_summarization and summarizer_tool is not None:
            objective_key = StreamlineManager._task_keys(tool_context).objective
            objective = tool_context.state.get(objective_key, "")
            payload = {
                "objective": objective,
                "records": mgr.get_records(tool_context),
                "result": result,
                "status": status,
            }
            sum_args = {"request": json.dumps(payload, ensure_ascii=False, indent=2)}
            raw = await summarizer_tool.run_async(
                args=sum_args, tool_context=tool_context
            )
            summary = (
                raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
            )

        mgr.finish(
            status=status,
            result=result,
            summary=summary,
            ctx=tool_context,
        )

        # Force quit
        tool_context._invocation_context.end_invocation = True
        return {"result": "ok", "instructions": "stop the execution now"}

    # ── Assemble tool list ──────────────────────────────────────────
    tools: list[Callable[..., Any]] = [
        add_subtask,
        get_current_subtask,
        list_subtasks,
        get_records,
        execute_current_subtask,
        decompose_subtask,
        finish,
    ]
    if use_skip:
        tools.append(skip)

    return tools
