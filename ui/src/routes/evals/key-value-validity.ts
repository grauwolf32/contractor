import { createContext, useCallback, useState } from "react";

export const KeyValueValidityContext = createContext<
  ((id: string, invalid: boolean) => void) | null
>(null);

export const KeyValueValidityProvider = KeyValueValidityContext.Provider;

export function useKeyValueValidity() {
  const [invalidEditors, setInvalidEditors] = useState<Set<string>>(
    () => new Set(),
  );
  const register = useCallback((id: string, invalid: boolean) => {
    setInvalidEditors((current) => {
      if (current.has(id) === invalid) return current;
      const next = new Set(current);
      if (invalid) next.add(id);
      else next.delete(id);
      return next;
    });
  }, []);
  return { invalid: invalidEditors.size > 0, register };
}
