import {
  useCallback,
  useContext,
  useEffect,
  useLayoutEffect,
  useRef,
} from "react";
import {
  UNSAFE_DataRouterContext,
  useLocation,
  type Location,
} from "react-router";

/**
 * Where the user is right now, while that is still this page. A late answer
 * (a recorded decision, a started check) asks before it navigates on the
 * user's behalf.
 *
 * React Router applies navigations in a transition, so a page the user is
 * leaving stays mounted — with its useLocation() unchanged — while the next
 * page's code loads and until React commits the new page. The returned
 * function reads the router itself instead: the destination of a navigation
 * in progress, otherwise its location. It returns undefined once that path
 * is no longer this page's, and after the page unmounted.
 */
export function usePageLocationNow(): () => Location | undefined {
  const location = useLocation();
  const dataRouter = useContext(UNSAFE_DataRouterContext);
  const rendered = useRef(location);
  const mounted = useRef(false);
  useLayoutEffect(() => {
    rendered.current = location;
  });
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);
  return useCallback(() => {
    if (!mounted.current) return undefined;
    const page = rendered.current;
    const state = dataRouter?.router.state;
    // Outside a data router only the rendered location is known.
    if (state === undefined) return page;
    const now = state.navigation.location ?? state.location;
    return now.pathname === page.pathname ? now : undefined;
  }, [dataRouter]);
}
