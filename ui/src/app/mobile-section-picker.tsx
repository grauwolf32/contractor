import { useNavigate } from "react-router";

export function MobileSectionPicker({
  label,
  value,
  options,
  state,
}: {
  label: string;
  value: string;
  options: readonly { to: string; label: string }[];
  state?: unknown;
}) {
  const navigate = useNavigate();
  return (
    <label className="mobile-section-picker">
      {label}
      <select
        aria-label={label}
        value={value}
        onChange={(event) =>
          void navigate(event.currentTarget.value, { state })
        }
      >
        {options.map((option) => (
          <option key={option.to} value={option.to}>
            {option.label}
          </option>
        ))}
      </select>
    </label>
  );
}
