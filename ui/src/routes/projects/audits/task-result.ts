/** A batch remains one immutable artifact. Select its trusted Task member only
 * for display; downloading still returns the original exact bytes. */
export function taskResultPreview(text: string, taskId: string): string {
  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch {
    return text.slice(0, 32768);
  }
  if (
    typeof value !== "object" ||
    value === null ||
    !("schema" in value) ||
    value.schema !== "contractor.audit.result.v1" ||
    !("items" in value) ||
    !Array.isArray(value.items)
  ) {
    return text.slice(0, 32768);
  }
  const items = value.items.filter(
    (item: unknown) =>
      typeof item === "object" &&
      item !== null &&
      "origin" in item &&
      typeof item.origin === "object" &&
      item.origin !== null &&
      "task_id" in item.origin &&
      item.origin.task_id === taskId,
  );
  if (items.length !== 1)
    throw new Error("Saved result has no unique member for this task");
  return JSON.stringify(items[0], null, 2).slice(0, 32768);
}
