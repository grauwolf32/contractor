import { createContext } from "react";

/**
 * Slot element to the right of the project section tabs. `undefined` means no
 * workspace shell is present (sections render their toolbar inline); `null`
 * means the shell exists but has not mounted its slot yet.
 */
export const ProjectSectionActionsContext = createContext<
  HTMLElement | null | undefined
>(undefined);
