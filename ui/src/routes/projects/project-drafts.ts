import { createContext, useContext, useEffect, useId } from "react";

/**
 * Collects unsaved drafts on the selected project's page (Projects route).
 * J / K would move to another project and discard them without a warning,
 * so the route turns them off while any draft is unsaved. Elsewhere (the
 * evaluation workspaces) there is no collector and reporting does nothing.
 */
export const ProjectDraftsContext = createContext<
  ((draft: string, unsaved: boolean) => void) | null
>(null);

/** Reports this component's draft as unsaved while `unsaved` is true. */
export function useUnsavedDraft(unsaved: boolean): void {
  const report = useContext(ProjectDraftsContext);
  const draft = useId();
  useEffect(() => {
    if (report === null || !unsaved) return undefined;
    report(draft, true);
    return () => report(draft, false);
  }, [draft, report, unsaved]);
}
