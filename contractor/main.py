import json

from contractor.runners.task_runner import TaskRunner, TaskRunnerEvent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="contractor",
        description="Run contractor task pipeline for a project",
    )

    parser.add_argument(
        "project_path",
        type=Path,
        help="Path to the project directory",
    )
    parser.add_argument(
        "--user-id",
        default="cli-user",
        help="User id for ADK session runner",
    )

    return parser.parse_args()


async def handle_event(event: TaskRunnerEvent) -> None:
    if event.type == "task_started":
        print(f"START TASK: {event.task_id} / {event.task_name}")

    elif event.type == "tool_call":
        print(f"[tool] {event.payload['tool_name']}")
        print(json.dumps(event.payload["tool_args"], ensure_ascii=False, indent=2))

    elif event.type == "final_text":
        print(f"[final-text] {event.task_name}")
        print(event.payload["text"])

    elif event.type == "iteration_result":
        print(
            json.dumps(
                {
                    "task": event.task_name,
                    "iteration": event.payload["iteration"],
                    "status": event.payload["status"],
                    "completed": event.payload["completed"],
                    "summary": event.payload["summary"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    elif event.type == "global_task_finished":
        print(f"END TASK: {event.task_name}")
        print(event.payload["summary"])


def get_oas_builder(project_path: str) -> TaskRunner:
    runner = TaskRunner(name="contractor")
    runner.add_task("dependency_information")


async def async_main():
    results = await runner.run(
        user_id="cli-user",
        on_event=handle_event,
    )

    print(json.dumps(results, ensure_ascii=False, indent=2))


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
