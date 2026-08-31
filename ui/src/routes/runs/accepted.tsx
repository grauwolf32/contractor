import { Link, useParams } from "react-router";

const RUN_ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$/;

export function AcceptedRunRoute() {
  const { runId = "" } = useParams();
  if (!RUN_ID_PATTERN.test(runId)) {
    return (
      <section className="route-page">
        <h2>Invalid Run route</h2>
        <Link to="/runs">Return to Runs</Link>
      </section>
    );
  }
  return (
    <section className="route-page accepted-run-page">
      <p className="eyebrow">Authoritative acceptance</p>
      <h2>Run accepted</h2>
      <div className="notice notice-success" role="status">
        <strong>Go Server accepted the Workflow Run.</strong>
        <code>{runId}</code>
      </div>
      <p className="lede">
        Live lifecycle inspection is delivered by the next UI slice. The Run ID
        above comes only from the Server's successful 202 response.
      </p>
      <Link to="/runs">Open the Runs workspace</Link>
    </section>
  );
}
