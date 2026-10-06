import {
  setThemePreference,
  THEME_PREFERENCES,
  type ThemePreference,
  useResolvedTheme,
  useThemePreference,
} from "../../app/theme";

const THEME_OPTIONS: Record<
  ThemePreference,
  { label: string; description: string }
> = {
  system: {
    label: "System",
    description: "Light or dark, following this device's setting.",
  },
  light: { label: "Light", description: "Light surfaces at any time of day." },
  dark: { label: "Dark", description: "Navy-charcoal surfaces." },
  black: {
    label: "Black",
    description: "Pure black, for OLED screens and dark rooms.",
  },
};

export function AppearanceSettings({ ordinal }: { ordinal: string }) {
  const preference = useThemePreference();
  const resolved = useResolvedTheme();
  return (
    <section
      id="appearance"
      className="settings-section"
      aria-labelledby="appearance-heading"
    >
      <header className="settings-section-header">
        <div className="settings-section-identity">
          <span className="settings-section-mark" aria-hidden="true">
            {ordinal}
          </span>
          <div>
            <p className="eyebrow">Appearance</p>
            <h3 id="appearance-heading">Theme</h3>
          </div>
        </div>
        <span className="settings-scope-badge">This browser</span>
      </header>

      <div className="settings-section-grid">
        <div className="settings-section-copy">
          <p>
            Choose how Contractor looks. The choice is saved in this browser
            only and applies to every open tab.
          </p>
        </div>

        <div className="settings-editor">
          <fieldset className="appearance-options">
            <legend className="visually-hidden">Theme</legend>
            {THEME_PREFERENCES.map((option) => (
              <label key={option} className="appearance-option">
                <input
                  type="radio"
                  name="theme"
                  value={option}
                  checked={preference === option}
                  onChange={() => setThemePreference(option)}
                />
                <span>
                  <strong>{THEME_OPTIONS[option].label}</strong>
                  <small>{THEME_OPTIONS[option].description}</small>
                </span>
              </label>
            ))}
          </fieldset>
          {preference === "system" ? (
            <p className="settings-related-link" role="status">
              Showing the {resolved} theme now.
            </p>
          ) : null}
        </div>
      </div>
    </section>
  );
}
