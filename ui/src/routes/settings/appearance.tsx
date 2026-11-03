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
  { label: string; description: string; className: string }
> = {
  system: {
    label: "System",
    className: "ops-theme-system",
    description: "Light or dark, following this device's setting.",
  },
  light: {
    className: "ops-theme-light",
    label: "Light",
    description: "Light surfaces at any time of day.",
  },
  dark: {
    className: "ops-theme-dark",
    label: "Dark",
    description: "Navy-charcoal surfaces.",
  },
  black: {
    label: "Black",
    className: "ops-theme-black",
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
            <span
              className={`ops-theme-swatch ${THEME_OPTIONS[option].className}`}
              aria-hidden="true"
            >
              <i />
              <i />
              <i />
            </span>
            <span>
              <strong>{THEME_OPTIONS[option].label}</strong>
              <small>{THEME_OPTIONS[option].description}</small>
            </span>
          </label>
        ))}
      </fieldset>
      {preference === "system" ? (
        <p className="ui-caption" role="status">
          Showing the {resolved} theme now.
        </p>
      ) : null}
    </SettingSection>
  );
}
