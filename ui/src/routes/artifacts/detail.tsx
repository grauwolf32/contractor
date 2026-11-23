import { Link, useParams, useSearchParams } from "react-router";

import {
  ARTIFACT_NAME_PATTERN,
  ARTIFACT_REVISION_PATTERN,
} from "../../api/artifacts";
import { ReturnLink } from "../../app/context-navigation";
import { useDocumentTitle } from "../../app/document-title";
import { ErrorNotice } from "../../app/error-notice";
import { ArtifactDetailView } from "./artifact-detail-view";
import "./materials.css";

const USER_SCOPE = { kind: "user" } as const;

/**
 * Library → Files → one file (`/artifacts/:namespace/:name[?revision=]`).
 * Skill packages open here too and return to Skills.
 */
export function ArtifactDetailRoute() {
  const { namespace = "", name = "" } = useParams();
  const [searchParams] = useSearchParams();
  const revision = searchParams.get("revision") ?? undefined;
  const skill = namespace === "skills";
  useDocumentTitle(
    name ? `${namespace}/${name} · ${skill ? "Skills" : "Files"}` : "Files",
  );
  const validIdentity =
    ARTIFACT_NAME_PATTERN.test(namespace) && ARTIFACT_NAME_PATTERN.test(name);
  const validRevision =
    revision === undefined || ARTIFACT_REVISION_PATTERN.test(revision);

  if (!validIdentity || !validRevision) {
    return (
      <section className="route-page materials-page materials-invalid">
        <ErrorNotice error={new Error("This file link is not valid.")} />
        <Link to="/artifacts">Return to Files</Link>
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
      variant="file"
      heading={
        <ReturnLink
          to={skill ? "/catalog/skills" : "/artifacts"}
          label={skill ? "Skills" : "Files"}
        />
      }
    />
  );
}
