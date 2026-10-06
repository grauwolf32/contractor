/** Project pages the Start page links to. */
export function projectPaths(projectId: string) {
  const root = `/projects/${encodeURIComponent(projectId)}`;
  return {
    overview: root,
    /** Project materials with the Add material dialog open. */
    addMaterial: `${root}/artifacts?add=artifact`,
    /** Project settings, where the live target is set. */
    settings: `${root}/settings`,
  };
}
