import { GitKeySettings } from "./git-key";

export function SettingsRoute() {
  return (
    <section className="route-page">
      <header className="section-heading">
        <div>
          <p className="eyebrow">Personal access</p>
          <h2>Settings</h2>
        </div>
      </header>
      <GitKeySettings />
    </section>
  );
}
