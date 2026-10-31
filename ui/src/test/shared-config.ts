import { createHash } from "node:crypto";
import { PublicAPI } from "../api/client";
import fixture from "../../../api/testdata/v1alpha1/valid/shared-config.json" with { type: "json" };
import type { SharedConfig } from "../api/shared-config";

/** The generated Server policy used by fake servers and isolated component tests. */
export const TEST_SHARED_CONFIG = fixture as SharedConfig;

/** A static test Server that answers startup config before delegating other calls. */
export class TestPublicAPI extends PublicAPI {
  constructor(
    ...[config, implementation = fetch]: ConstructorParameters<typeof PublicAPI>
  ) {
    super(config, async (input) => {
      const request = input instanceof Request ? input : new Request(input);
      if (
        request.method === "GET" &&
        new URL(request.url).pathname === "/v1/config"
      ) {
        return new Response(JSON.stringify(TEST_SHARED_CONFIG), {
          headers: {
            "content-type": "application/json",
            "X-Contractor-API-Version": "contractor.public.v1",
          },
        });
      }
      return implementation(input);
    });
  }
}

/** A different valid Server policy for boundary tests, without changing production defaults. */
export function testSharedConfig(
  artifactPayloadMaxBytes: number,
): SharedConfig {
  const content = { limits: { artifactPayloadMaxBytes }, version: 1 as const };
  const digest = createHash("sha256")
    .update(JSON.stringify(content))
    .digest("hex");
  return {
    apiVersion: "contractor/v1alpha1",
    ...content,
    digest: `sha256:${digest}`,
  };
}
