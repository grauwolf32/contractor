import { dereference, validate, type Schema } from "@cfworker/json-schema";
import { parseDocument } from "yaml";
import schema from "../../../../../api/v1alpha1/engagement.schema.json";
import { MAXIMUM_PREVIEW_BYTES } from "../../../api/artifacts";

const lookup = dereference(schema as Schema);
export const ENGAGEMENT_METHODS =
  schema.$defs.Route.properties.methods.items.enum;

/** A display projection. The shared schema owns structure, enums and bounds;
 * Server admission owns canonical network grammar and cross-field authority. */
export interface EngagementDocument {
  schema: string;
  egress: { routes: EngagementRoute[]; exclusions: EngagementRoute[] };
  identities: {
    id: string;
    kind: string;
    origins: string[];
    role?: string;
    tenant?: string;
    credential?: { credentialId: string; kind: string };
  }[];
  limits?: {
    requestsPerSecond?: number;
    concurrentRequests?: number;
    requestBodyBytes?: number;
    responseBodyBytes?: number;
  };
}
export interface EngagementRoute {
  origin: string;
  methods: string[];
  pathPrefixes: string[];
  addressRanges: string[];
}

export function parseEngagementDocument(
  source: string,
): EngagementDocument | undefined {
  if (new TextEncoder().encode(source).length > MAXIMUM_PREVIEW_BYTES)
    return undefined;
  try {
    const value: unknown = JSON.parse(source);
    if (
      parseDocument(source, { uniqueKeys: true, prettyErrors: false }).errors
        .length
    )
      return undefined;
    if (!validate(value, schema as Schema, "2020-12", lookup, true).valid)
      return undefined;
    return value as EngagementDocument;
  } catch {
    return undefined;
  }
}
