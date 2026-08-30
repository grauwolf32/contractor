import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  createContext,
  type ReactNode,
  useCallback,
  useContext,
  useMemo,
} from "react";

import type { AuthSession, LoginRequest, PublicAPI } from "../api/client";
import { queryKeys } from "../api/query-keys";

interface SessionContextValue {
  session: AuthSession | null | undefined;
  error: Error | null;
  isLoading: boolean;
  isLoggingIn: boolean;
  isLoggingOut: boolean;
  login: (request: LoginRequest) => Promise<AuthSession>;
  logout: () => Promise<void>;
}

const SessionContext = createContext<SessionContextValue | null>(null);

export type SessionAPI = Pick<PublicAPI, "getSession" | "login" | "logout">;

export function SessionProvider({
  api,
  children,
}: {
  api: SessionAPI;
  children: ReactNode;
}) {
  const queryClient = useQueryClient();
  const sessionQuery = useQuery({
    queryKey: queryKeys.session,
    queryFn: () => api.getSession(),
    retry: false,
    staleTime: 30_000,
  });
  const loginMutation = useMutation({
    mutationFn: (body: LoginRequest) => api.login(body),
  });
  const logoutMutation = useMutation({ mutationFn: () => api.logout() });

  const login = useCallback(
    async (body: LoginRequest) => {
      const session = await loginMutation.mutateAsync(body);
      queryClient.setQueryData(queryKeys.session, session);
      return session;
    },
    [loginMutation, queryClient],
  );
  const logout = useCallback(async () => {
    try {
      await logoutMutation.mutateAsync();
      queryClient.setQueryData(queryKeys.session, null);
    } catch (error) {
      await queryClient.invalidateQueries({ queryKey: queryKeys.session });
      throw error;
    }
  }, [logoutMutation, queryClient]);

  const value = useMemo<SessionContextValue>(
    () => ({
      session: sessionQuery.data,
      error: sessionQuery.error,
      isLoading: sessionQuery.isPending,
      isLoggingIn: loginMutation.isPending,
      isLoggingOut: logoutMutation.isPending,
      login,
      logout,
    }),
    [
      sessionQuery.data,
      sessionQuery.error,
      sessionQuery.isPending,
      loginMutation.isPending,
      logoutMutation.isPending,
      login,
      logout,
    ],
  );

  return (
    <SessionContext.Provider value={value}>{children}</SessionContext.Provider>
  );
}

export function useSession(): SessionContextValue {
  const value = useContext(SessionContext);
  if (value === null) {
    throw new Error("useSession must be used inside SessionProvider");
  }
  return value;
}
