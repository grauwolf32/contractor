import { useContext, type ComponentProps } from "react";
import { ContextLink } from "../../app/context-navigation";
import { RunListSourceContext } from "./source-context";

/** Add the current list's read descriptor, never its business records. */
export function RunSourceLink(props: ComponentProps<typeof ContextLink>) {
  const source = useContext(RunListSourceContext);
  return (
    <ContextLink
      {...props}
      contextState={source === undefined ? undefined : { runSource: source }}
    />
  );
}
