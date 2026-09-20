import { Link, useLocation } from "react-router";
import "./configuration-reading.css";

const destinations = [
  { to: "/runs/configuration", label: "Runtime defaults & labels" },
  { to: "/operations/configurations", label: "Models & gateways" },
  { to: "/operations/credentials", label: "LLM credentials & budgets" },
  { to: "/operations/settings", label: "Scheduler & Git access" },
];

export function ConfigurationLinks() {
  const { pathname } = useLocation();
  return (
    <nav
      className="configuration-scope-links"
      aria-label="Related configuration"
    >
      <span className="configuration-scope-label">Related settings</span>
      {destinations
        .filter(({ to }) => pathname !== to && !pathname.startsWith(`${to}/`))
        .map(({ to, label }) => (
          <Link key={to} to={to}>
            {label}
          </Link>
        ))}
    </nav>
  );
}
