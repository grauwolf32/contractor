import { ENGAGEMENT_MEDIA_TYPE } from "../../api/artifacts";

export const workflowFormats: Record<string, string> = {
  [ENGAGEMENT_MEDIA_TYPE]: "Engagement",
  "application/json": "JSON",
  "application/zip": "ZIP",
  "application/vnd.contractor.agent-skill+zip": "Skill package",
  "application/yaml": "YAML",
  "text/plain": "Text",
  "text/vnd.contractor.wordlist": "Wordlist",
  "text/markdown": "Markdown",
  "text/vnd.likec4": "LikeC4",
  "text/x-diff": "Diff",
  "application/vnd.contractor.workspace-overlay+json": "Workspace overlay",
  "application/vnd.contractor.findings-collection+zip": "Findings ZIP",
};
