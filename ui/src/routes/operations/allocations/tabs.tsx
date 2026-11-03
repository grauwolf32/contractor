import { NavLink } from "react-router";

export function AllocationViewTabs() {
  return (
    <nav className="ops-subtabs" aria-label="Allocation views">
      <NavLink to="/operations/allocations" end>
        Current
      </NavLink>
      <NavLink to="/operations/allocations/completed">Completed</NavLink>
    </nav>
  );
}
