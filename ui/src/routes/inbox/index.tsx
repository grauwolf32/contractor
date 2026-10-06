import { useDocumentTitle } from "../../app/document-title";

/** Stub registered by the V3B foundation; the Inbox area replaces it. */
export function InboxRoute() {
  useDocumentTitle("Inbox");
  return (
    <section className="ui-stub">
      <h1>Inbox</h1>
      <p>This page is being built.</p>
    </section>
  );
}
