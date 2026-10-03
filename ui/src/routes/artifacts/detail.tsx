import { Link, useParams, useSearchParams } from "react-router";

import {
  ARTIFACT_NAME_PATTERN,
  ARTIFACT_REVISION_PATTERN,
} from "../../api/artifacts";
import { ReturnLink } from "../../app/context-navigation";
import { useDocumentTitle } from "../../app/document-title";
import { ErrorNotice } from "../../app/error-notice";
import { ArtifactDetailView } from "./artifact-detail-view";

const USER_SCOPE = { kind: "user" } as const;

export function ArtifactDetailRoute() {
  const { namespace = "", name = "" } = useParams();
  const [searchParams] = useSearchParams();
  const revision = searchParams.get("revision") ?? undefined;
  useDocumentTitle(name ? `${namespace}/${name}` : "Artifact");
  const validIdentity =
    ARTIFACT_NAME_PATTERN.test(namespace) && ARTIFACT_NAME_PATTERN.test(name);
  const validRevision =
    revision === undefined || ARTIFACT_REVISION_PATTERN.test(revision);

  if (!validIdentity || !validRevision) {
    return (
      <section className="route-page">
        <ErrorNotice error={new Error("Artifact route is invalid")} />
        <Link to="/artifacts">Return to Artifacts</Link>
      </section>
    );
  }

  return (
    <ArtifactDetailView
      key={`${namespace}/${name}`}
      scope={USER_SCOPE}
      namespace={namespace}
      name={name}
      revision={revision}
      heading={
        <ReturnLink
          to={namespace === "skills" ? "/catalog/skills" : "/artifacts"}
          label={namespace === "skills" ? "Skills" : "All Artifacts"}
        />
      }
    />
  );
}
