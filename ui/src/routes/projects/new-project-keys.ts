import { useState } from "react";

import type { CreateProjectRequest } from "../../api/projects";
import { MutationDraftKeyring } from "../../mutations/idempotency";

/**
 * The Idempotency-Key memory of New project. The page that opens the dialog
 * keeps it (useNewProjectKeys), so it outlives the dialog: after a lost
 * response, closing and reopening the dialog and sending the same name and
 * description reuses the key, and the Server cannot create a second project.
 * Once a project is created the memory starts over, so creating another
 * project with the same values is a new request.
 */
export class NewProjectKeys {
  #keyring = new MutationDraftKeyring<CreateProjectRequest>("create-project");

  /** The key for this request: the same for as long as it is unchanged. */
  keyFor(request: CreateProjectRequest): string {
    return this.#keyring.keyFor(request);
  }

  /** The Server created the project: forget the request. */
  created(): void {
    this.#keyring = new MutationDraftKeyring<CreateProjectRequest>(
      "create-project",
    );
  }
}

/** New project keys that live as long as the calling page. */
export function useNewProjectKeys(): NewProjectKeys {
  const [keys] = useState(() => new NewProjectKeys());
  return keys;
}
