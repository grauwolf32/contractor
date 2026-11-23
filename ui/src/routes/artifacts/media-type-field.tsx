import { workflowFormats } from "../workflows/formats";

export function ArtifactMediaTypeField({
  value,
  disabled = false,
  onChange,
}: {
  value: string;
  disabled?: boolean;
  onChange: (mediaType: string) => void;
}) {
  return (
    <>
      <label>
        File format
        <select
          disabled={disabled}
          value={Object.hasOwn(workflowFormats, value) ? value : ""}
          onChange={(event) => onChange(event.target.value)}
        >
          <option value="" disabled>
            Custom media type
          </option>
          {Object.entries(workflowFormats).map(([mediaType, label]) => (
            <option key={mediaType} value={mediaType}>
              {label}
            </option>
          ))}
        </select>
      </label>
      <label>
        Media type
        <input
          name="mediaType"
          required
          disabled={disabled}
          value={value}
          onChange={(event) => onChange(event.target.value)}
        />
      </label>
    </>
  );
}
