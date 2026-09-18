import { Link, useLocation, type LinkProps } from "react-router";
import { catalogReturnState, locationDestination } from "./navigation";

export function ContextLink({
  returnLabel,
  returnHash,
  ...props
}: Omit<LinkProps, "state"> & { returnLabel: string; returnHash?: string }) {
  const location = useLocation();
  return (
    <Link
      {...props}
      state={{
        returnTo:
          returnHash === undefined
            ? locationDestination(location)
            : `${location.pathname}${location.search}${returnHash}`,
        returnLabel,
        returnState: location.state,
      }}
    />
  );
}

export function ReturnLink({ to, label }: { to: string; label: string }) {
  const { state } = useLocation();
  const destination = catalogReturnState(state, {
    returnTo: to,
    returnLabel: label,
  });
  return (
    <Link
      className="back-link"
      to={destination.returnTo}
      state={destination.returnState}
    >
      ← {destination.returnLabel}
    </Link>
  );
}
