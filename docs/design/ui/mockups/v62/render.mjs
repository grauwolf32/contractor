// Render the standalone design alternatives; no Server, database or model calls.
import { execFileSync } from "node:child_process";
import {
  mkdtempSync,
  mkdirSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const directory = dirname(fileURLToPath(import.meta.url));
const output = resolve(directory, "../../screenshots/v62");
const browser =
  process.env.FIREFOX_BIN ||
  (process.platform === "darwin"
    ? "/Applications/Firefox.app/Contents/MacOS/firefox"
    : "firefox");
const variants = { a: "a-overview", b: "b-workspace", c: "c-coverage" };
const cases = [];
for (const [variant, stem] of Object.entries(variants)) {
  for (const theme of ["light", "dark", "black"]) {
    cases.push({
      variant,
      theme,
      state: "running",
      name: stem + (theme === "light" ? "" : `-${theme}`),
      width: 1440,
      height: 1180,
    });
  }
  cases.push({
    variant,
    theme: "light",
    state: "running",
    name: `${stem}-mobile`,
    width: 390,
    height: 1900,
  });
}
cases.push(
  {
    variant: "a",
    theme: "light",
    state: "preparing",
    name: "a-preparing",
    width: 1440,
    height: 1060,
  },
  {
    variant: "a",
    theme: "light",
    state: "failed",
    name: "a-failed",
    width: 1440,
    height: 1060,
  },
  {
    variant: "a",
    theme: "light",
    state: "running",
    mode: "url",
    name: "a-url-checks",
    width: 1440,
    height: 1180,
  },
);
mkdirSync(output, { recursive: true });
const profile = mkdtempSync(join(tmpdir(), "contractor-v62-firefox-"));
writeFileSync(
  join(profile, "user.js"),
  [
    'user_pref("browser.shell.checkDefaultBrowser", false);',
    'user_pref("browser.aboutwelcome.enabled", false);',
    'user_pref("browser.startup.homepage_override.mstone", "ignore");',
    'user_pref("datareporting.policy.dataSubmissionEnabled", false);',
    'user_pref("toolkit.telemetry.enabled", false);',
  ].join("\n"),
);
try {
  console.log(
    execFileSync(browser, ["--version"], { encoding: "utf8" }).trim(),
  );
  for (const item of cases) {
    const url = pathToFileURL(join(directory, `${item.variant}.html`));
    url.search = new URLSearchParams({
      state: item.state,
      theme: item.theme,
      ...(item.mode ? { mode: item.mode } : {}),
    });
    const path = join(output, `${item.name}.png`);
    execFileSync(
      browser,
      [
        "--headless",
        "--new-instance",
        "--profile",
        profile,
        "--window-size",
        `${item.width},${item.height}`,
        "--screenshot",
        path,
        url.href,
      ],
      { timeout: 30000, stdio: "pipe" },
    );
    const bytes = readFileSync(path);
    if (
      bytes.readUInt32BE(16) !== item.width ||
      bytes.readUInt32BE(20) !== item.height
    ) {
      throw new Error(`Unexpected render dimensions: ${item.name}`);
    }
    console.log(`${item.name}.png · ${item.width} × ${item.height}`);
  }
} finally {
  rmSync(profile, { recursive: true, force: true });
}
