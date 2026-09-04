import type { ArtifactMetadata } from "../../api/artifacts";
import type { WorkflowSummary } from "../../api/workflows";
import {
  artifactAccepts,
  artifactOptionKey,
} from "../../run-drafts/validation";

export interface WorkflowCompatibility {
  workflow: WorkflowSummary;
  candidates: Readonly<Record<string, readonly ArtifactMetadata[]>>;
  preselected: Readonly<Record<string, string>>;
  missingRequiredInputs: readonly string[];
  primaryOutputs: readonly string[];
  compatible: boolean;
  suppressed: boolean;
}

function workflowSelector(workflow: WorkflowSummary): string {
  return `${workflow.ref.name}@${workflow.ref.version}`;
}

function artifactIdentity(metadata: ArtifactMetadata): string {
  return artifactOptionKey(metadata.artifact);
}

export function buildWorkflowCompatibility(
  workflows: readonly WorkflowSummary[],
  artifacts: readonly ArtifactMetadata[],
): WorkflowCompatibility[] {
  const currentArtifacts = artifacts
    .filter((artifact) => artifact.current)
    .sort((left, right) =>
      artifactIdentity(left).localeCompare(artifactIdentity(right)),
    );
  const outputBindings = new Set(
    currentArtifacts
      .filter((artifact) => artifact.artifact.namespace === "outputs")
      .map((artifact) => artifact.artifact.name),
  );

  return [...workflows]
    .map((workflow): WorkflowCompatibility => {
      const candidates: Record<string, readonly ArtifactMetadata[]> = {};
      const preselected: Record<string, string> = {};
      const missingRequiredInputs: string[] = [];
      for (const [slotName, slot] of Object.entries(workflow.inputs ?? {}).sort(
        ([left], [right]) => left.localeCompare(right),
      )) {
        const accepted = currentArtifacts.filter((artifact) =>
          artifactAccepts(slot.mediaTypes, artifact),
        );
        candidates[slotName] = accepted;
        if (accepted.length === 1) {
          preselected[slotName] = artifactOptionKey(accepted[0]!.artifact);
        }
        if (slot.required && accepted.length === 0) {
          missingRequiredInputs.push(slotName);
        }
      }
      const primaryOutputs = Object.entries(workflow.outputs ?? {})
        .filter(([, output]) => output.primary === true)
        .map(([name]) => name)
        .sort();
      const compatible = missingRequiredInputs.length === 0;
      return {
        workflow,
        candidates,
        preselected,
        missingRequiredInputs,
        primaryOutputs,
        compatible,
        suppressed:
          compatible &&
          primaryOutputs.length > 0 &&
          primaryOutputs.every((name) => outputBindings.has(name)),
      };
    })
    .sort((left, right) =>
      workflowSelector(left.workflow).localeCompare(
        workflowSelector(right.workflow),
      ),
    );
}
