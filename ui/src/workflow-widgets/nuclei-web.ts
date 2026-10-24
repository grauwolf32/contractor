import type { WorkflowResource } from "../api/workflows";

// This widget owns the Nuclei-specific fields. Launching Runs, artifact
// transport, Project selection and sweep progress remain in their shared hosts.
export const nucleiWebWidget = {
  id: "nuclei-web",
  title: "Web scan",
  path: "/checks/web/new",
  workflow: { name: "nuclei-web", version: "1" },
  fields: [
    {
      name: "target",
      label: "Target URL",
      hint: "The HTTP or HTTPS address this scan will contact.",
      required: true,
      type: "url",
    },
    {
      name: "template_ids",
      label: "Rule IDs",
      hint: "Comma-separated Nuclei rule IDs or patterns. Empty selects all matching rules.",
    },
    { name: "tags", label: "Tags", hint: "Optional comma-separated tags." },
    {
      name: "severity",
      label: "Severity",
      hint: "Optional: info, low, medium, high, critical or unknown (comma-separated).",
    },
    {
      name: "rate_limit",
      label: "Requests per second",
      hint: "Default 10. Enter a whole number from 1 to 1000.",
      placeholder: "10",
      inputMode: "numeric",
    },
  ],
  supports(workflow: WorkflowResource): boolean {
    const parameters = Object.keys(workflow.parameters).sort();
    const expected = this.fields.map((field) => field.name).sort();
    const pack = workflow.inputs.rulepack;
    return (
      workflow.ref.name === this.workflow.name &&
      workflow.ref.version === this.workflow.version &&
      parameters.length === expected.length &&
      parameters.every(
        (name, i) =>
          name === expected[i] &&
          workflow.parameters[name]?.required === (name === "target"),
      ) &&
      Object.keys(workflow.inputs).length === 1 &&
      pack !== undefined &&
      !pack.required &&
      pack.sourceScope === "user" &&
      pack.mediaTypes.length > 0 &&
      pack.mediaTypes.every((type) =>
        [
          "application/yaml",
          "application/vnd.contractor.nuclei-rulepack+zip",
        ].includes(type),
      )
    );
  },
};
