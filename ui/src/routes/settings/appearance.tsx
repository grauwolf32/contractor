import {
  setThemePreference,
  THEME_PREFERENCES,
  type ThemePreference,
  useResolvedTheme,
  useThemePreference,
} from "../../app/theme";
import { SettingSection, type SettingHeadingLevel } from "./section";

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

export function AppearanceSettings({
  titleAs = "h3",
}: {
  titleAs?: SettingHeadingLevel | undefined;
}) {
  const preference = useThemePreference();
  const resolved = useResolvedTheme();
  return (
    <SettingSection
      id="appearance"
      titleAs={titleAs}
      eyebrow="Appearance"
      title="Theme"
      scope="This browser"
      about={
        <p>
          Choose how Contractor looks. The choice is saved in this browser only
          and applies to every open tab.
        </p>
      }
    >
      <fieldset className="ops-theme-options">
        <legend className="ui-visually-hidden">Theme</legend>
        {THEME_PREFERENCES.map((option) => (
          <label key={option} className="ops-theme-option">
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
        <p className="ops-note" role="status">
          Showing the {resolved} theme now.
        </p>
      ) : null}
    </SettingSection>
  );
}
