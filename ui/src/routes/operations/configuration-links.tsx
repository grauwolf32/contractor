import { NavLink } from "react-router";
import "./configuration-reading.css";

export function ConfigurationLinks() {
  return (
    <nav
      className="configuration-scope-links"
      aria-label="Related configuration"
    >
      <NavLink to="/runs/configuration">Runtime defaults & labels</NavLink>
      <NavLink to="/operations/configurations">Models & gateways</NavLink>
      <NavLink to="/operations/credentials">LLM credentials & budgets</NavLink>
      <NavLink to="/operations/settings">Scheduler & Git access</NavLink>
    </nav>
  );
}
