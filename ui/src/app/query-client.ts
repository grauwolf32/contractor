import { QueryClient } from "@tanstack/react-query";
import { APICompatibilityError, PublicAPIError } from "../api/error";

/** Session recovery and shared policy bootstrap use one bounded retry rule. */
export const startupQueryRetry = {
  retry: (failures: number, error: Error): boolean =>
    failures < 3 &&
    error instanceof PublicAPIError &&
    !(error instanceof APICompatibilityError) &&
    error.retryable,
  retryDelay: (attempt: number): number => Math.min(200 * 2 ** attempt, 800),
};

export function createApplicationQueryClient(): QueryClient {
  return new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        refetchOnWindowFocus: false,
      },
      mutations: { retry: false },
    },
  });
}
