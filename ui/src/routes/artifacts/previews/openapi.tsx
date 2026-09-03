import { ApiReferenceReact } from "@scalar/api-reference-react";
import "@scalar/api-reference-react/style.css";

const rejectNetworkRequest: typeof fetch = async () => {
  throw new TypeError("Network access is disabled in Artifact preview");
};

export default function OpenApiArtifactPreview({
  document,
}: {
  document: Record<string, unknown>;
}) {
  return (
    <div>
      <div className="artifact-renderer-toolbar">
        <span>Read-only · external references and requests are disabled</span>
      </div>
      <div className="openapi-artifact-preview">
        <ApiReferenceReact
          configuration={{
            content: document,
            agent: { disabled: true, hideAddApi: true },
            mcp: { disabled: true },
            customFetch: rejectNetworkRequest,
            documentDownloadType: "none",
            forceDarkModeState: "dark",
            hiddenClients: true,
            hideClientButton: true,
            hideDarkModeToggle: true,
            hideTestRequestButton: true,
            persistAuth: false,
            showDeveloperTools: "never",
            telemetry: false,
            theme: "default",
            withDefaultFonts: false,
          }}
        />
      </div>
    </div>
  );
}
