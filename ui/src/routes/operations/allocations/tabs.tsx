import { NavLink } from "react-router";

export function AllocationViewTabs() {
  return (
    <nav className="allocation-view-tabs" aria-label="Allocation views">
      <NavLink to="/operations/allocations" end>
        <strong>Current</strong>
        <small>Live slots</small>
      </NavLink>
      <NavLink to="/operations/allocations/completed">
        <strong>Completed</strong>
        <small>Resource history</small>
      </NavLink>
    </nav>
  );
}
