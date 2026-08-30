const copy = {
  workflows: {
    eyebrow: "Definition library",
    title: "Workflows",
    description:
      "Choose a versioned workflow and bind its parameters and artifacts.",
  },
  artifacts: {
    eyebrow: "Inputs and results",
    title: "Artifacts",
    description:
      "Upload user-scoped inputs and inspect exact immutable artifact versions.",
  },
  runs: {
    eyebrow: "Execution history",
    title: "Runs",
    description:
      "Follow stages, planner progress, frozen outputs, and terminal decisions.",
  },
  operations: {
    eyebrow: "Runtime overview",
    title: "Operations",
    description:
      "Inspect runtime agents, allocations, and published LLM configuration.",
  },
} as const;

export function PlaceholderRoute({ kind }: { kind: keyof typeof copy }) {
  const content = copy[kind];
  return (
    <section className="route-page">
      <header>
        <p className="eyebrow">{content.eyebrow}</p>
        <h2>{content.title}</h2>
        <p className="lede">{content.description}</p>
      </header>
      <div className="empty-state">
        <span>Foundation ready</span>
        <p>This domain view is delivered by the next granular UI task.</p>
      </div>
    </section>
  );
}

export function NotFoundRoute() {
  return (
    <main className="centered-state">
      <p className="eyebrow">404</p>
      <h1>That UI route does not exist</h1>
      <a href="/workflows">Return to workflows</a>
    </main>
  );
}
