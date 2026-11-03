import { chromium, expect } from '../../../../../ui/node_modules/@playwright/test/index.mjs';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import './prepare.mjs';

const root = new URL('./', import.meta.url);
const repo = new URL('../../../../../', root);
const origin = process.env.OPERATIONS_UI_ORIGIN ?? 'http://127.0.0.1:17560';
const route = origin + '/test-results/operations-production/current.html';
const manifest = JSON.parse(await readFile(new URL('reference-manifest.json', root), 'utf8'));
for (const [path, digest] of Object.entries(manifest.files)) {
  const actual = createHash('sha256').update(await readFile(new URL(path, repo))).digest('hex');
  if (actual !== digest) throw Error('Native reference changed: ' + path);
}
const headings = {
  overview: 'Execution slots',
  'runtime-agents': 'Runtime Agents',
  allocations: 'Current allocations',
  completed: 'Completed allocation resources',
  performance: 'Server performance',
  configuration: 'Runtime configuration',
  llm: 'LLM configurations',
  credentials: 'Managed LLM credentials',
  settings: 'Settings',
};
const errors = [], networkViolations = [], layouts = [], controls = [];
const browser = await chromium.launch();
await mkdir(new URL('review/', root), { recursive: true });
const context = await browser.newContext();
context.setDefaultTimeout(8000);
await context.route('**/*', async request => {
  const url = new URL(request.request().url());
  if (url.origin !== origin || url.pathname.startsWith('/v1/')) {
    networkViolations.push({ origin: url.origin, path: url.pathname, method: request.request().method() });
    await request.abort();
  } else await request.continue();
});
const page = await context.newPage();
page.on('pageerror', error => errors.push(error.message));
await page.clock.setFixedTime(new Date('2026-11-03T12:00:30Z'));
async function open(section = 'overview', width = 1440, theme = 'dark', scenario = 'normal') {
  await page.setViewportSize({ width, height: 1000 });
  await page.goto(`${route}?section=${section}&case=${scenario}&theme=${theme}`);
  await expect(page.getByRole('heading', { name: 'Operations', exact: true })).toBeVisible();
  if (scenario !== 'error') {
    await expect(page.getByRole('heading', { name: headings[section], exact: true })).toBeVisible();
    await expect(page.locator('.ops-page-body')).not.toContainText(/Loading [^\n]*…/);
  } else await expect(page.getByText('Public API is unavailable', { exact: true }).first()).toBeVisible();
  await page.evaluate(async () => { await document.fonts.ready; await new Promise(requestAnimationFrame); });
  if (scenario !== 'error') await expect(page.locator('.notice-error')).toHaveCount(0);
  if (section === 'performance' && scenario !== 'error' && scenario !== 'empty') await expect(page.locator('.ops-bar-chart')).toHaveCount(2);
  await expect(page.locator('body')).not.toContainText(/invalid API response|The Server sent an answer this page cannot read|Something went wrong|Unexpected Application Error/i);
}
async function measure(section, width, theme, scenario) {
  const metrics = await page.evaluate(() => ({
    width: innerWidth,
    content: document.documentElement.scrollWidth,
    font: getComputedStyle(document.body).fontFamily,
    theme: document.documentElement.dataset.theme,
    sidebar: getComputedStyle(document.querySelector('.ops-tabs')).display,
  }));
  if (metrics.content > width || !metrics.font.includes('system-ui') || metrics.theme !== theme) {
    throw Error('Layout drift: ' + JSON.stringify({ section, scenario, ...metrics }));
  }
  if (width <= 820) await expect(page.locator('.ops-mobile-navigation > summary')).toBeVisible();
  else await expect(page.getByRole('navigation', { name: 'Operations sections', exact: true })).toBeVisible();
  layouts.push({ section, scenario, ...metrics });
}
try {
  for (const section of process.env.OPERATIONS_UI_CONTROLS_ONLY ? [] : Object.keys(headings)) {
    for (const theme of ['light', 'dark', 'black']) for (const width of [1440, 1100, 390, 320]) {
      await open(section, width, theme, section === 'overview' ? 'attention' : 'normal');
      await measure(section, width, theme, section === 'overview' ? 'attention' : 'normal');
      if (theme === 'dark' && [1440, 390].includes(width)) {
        await page.screenshot({ path: new URL(`review/${section}-${theme}-${width}.png`, root).pathname, fullPage: true });
      }
    }
    console.log('Responsive production route passed: ' + section);
  }
  for (const section of ['overview', 'settings', 'performance', 'completed', 'configuration']) {
    for (const scenario of ['empty', 'error']) {
      await open(section, 390, 'black', scenario);
      await measure(section, 390, 'black', scenario);
    }
  }
  await open('overview', 390, 'dark', 'reconnecting');
  await expect(page.locator('.ops-page-actions')).toContainText('reconnecting');
  await measure('overview', 390, 'dark', 'reconnecting');

  // Exact snapshot counts, qualified reservation and lost-lease facts.
  await open('overview', 1440, 'dark', 'attention');
  await expect(page.getByRole('list', { name: 'Slots by state' }).locator('li')).toHaveText(['Idle 1', 'Busy 2', 'Reserved 1', 'Draining 1', 'Fenced 1']);
  const legend = await page.locator('.ops-legend').evaluate(el => ({ gap: getComputedStyle(el).columnGap, itemGaps: [...el.children].map(li => getComputedStyle(li).gap) }));
  if (legend.gap !== '8px' || legend.itemGaps.some(gap => gap !== '4px')) throw Error('Slot legend lost compact spacing');
  await page.getByRole('region', { name: 'Needs inspection' }).getByRole('link', { name: 'runtime-04', exact: true }).click();
  await expect(page.locator('#process-runtime-04')).toHaveAttribute('open');
  await expect(page.locator('#process-runtime-04')).toContainText('Not observed');
  await page.locator('#process-runtime-06 > summary').click();
  await expect(page.locator('#process-runtime-06')).toContainText('alloc-observed-06');
  await expect(page.locator('#process-runtime-06')).toContainText('alloc-06');
  controls.push('snapshot counts, compact state colors, inspection hash, full observed/authoritative process facts');

  for (const [label, section] of [['Settings', 'settings'], ['Configuration', 'configuration'], ['Runtime Agents', 'runtime-agents'], ['Performance', 'performance']]) {
    await page.getByRole('navigation', { name: 'Operations sections', exact: true }).getByRole('link', { name: label, exact: true }).click();
    await expect(page.getByRole('heading', { name: headings[section], exact: true })).toBeVisible();
  }
  await open('overview', 390);
  await page.locator('.ops-mobile-navigation > summary').click();
  await page.getByRole('navigation', { name: 'Operations mobile sections', exact: true }).getByRole('link', {name: 'Settings', exact: true}).click();
  await expect(page.locator('.ops-mobile-navigation')).not.toHaveAttribute('open');
  await expect(page.getByRole('heading', { name: 'Workflow scheduling', exact: true })).toBeVisible();
  controls.push('actual desktop routes and native mobile section navigation');

  await open('completed', 390);
  await page.getByRole('button', { name: 'Metrics for alloc-history-1', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Metrics for alloc-history-1', exact: true })).toHaveAttribute('aria-expanded', 'true');
  for (const state of ['available', 'partial', 'disabled', 'unavailable', 'unsupported', 'pending']) {
    await expect(page.locator('.ops-entry-button').getByText(state, { exact: true })).toBeVisible();
  }
  controls.push('actual completed allocation metric disclosure; six resource-collection outcomes');

  await open('configuration', 390);
  await page.locator('.ops-resource-disclosure > summary').first().click();
  await page.getByRole('button', { name: 'Manage bindings for default@1', exact: true }).click();
  await expect(page.getByRole('dialog', { name: 'Runtime label bindings', exact: true })).toBeVisible();
  await page.keyboard.press('Escape');
  await page.getByRole('button', { name: 'Publish RuntimeConfig', exact: true }).click();
  await expect(page.getByRole('dialog', { name: 'Publish RuntimeConfig', exact: true })).toBeVisible();
  await page.keyboard.press('Escape');
  controls.push('actual RuntimeConfig publication and label binding dialogs remain reachable');

  await open('llm', 390);
  await page.getByRole('group', { name: 'Kind', exact: true }).getByRole('button', { name: 'LLM gateways', exact: true }).click();
  await expect(page.getByText('local-litellm@1', { exact: true })).toBeVisible();
  await page.getByText('local-litellm@1', {exact:true}).click();
  await expect(page.getByRole('link', { name: 'Inspect / clone', exact: true })).toBeVisible();
  controls.push('actual LLM kind filter and configuration destination');

  await open('settings', 390);
  const modelSave = page.getByRole('button', { name: 'Save model defaults', exact: true });
  const scheduleSave = page.getByRole('button', { name: 'Save scheduling limits', exact: true });
  const modelLimit = page.getByLabel('Concurrent Runs using a model', { exact: true });
  const toolLimit = page.getByLabel('Concurrent Runs without a model', { exact: true });
  const up = page.getByRole('button', { name: 'Increase concurrent runs using a model', exact: true });
  const down = page.getByRole('button', { name: 'Decrease concurrent runs using a model', exact: true });
  await expect(modelSave).toHaveCount(0);
  await expect(scheduleSave).toHaveCount(0);
  await expect(down).toBeDisabled();
  await up.click();
  await expect(modelLimit).toHaveValue('2');
  await expect(modelSave).toHaveCount(0);
  await scheduleSave.click();
  await expect(page.getByText('Saved 2 using a model and 2 without a model.', { exact: true })).toBeVisible();
  await expect(scheduleSave).toHaveCount(0);
  await modelLimit.fill('32');
  await expect(up).toBeDisabled();
  await modelLimit.fill('33');
  await expect(scheduleSave).toBeDisabled();
  await page.getByRole('button', { name: 'Reset to saved values', exact: true }).click();
  await expect(modelLimit).toHaveValue('2');
  await expect(toolLimit).toHaveValue('2');
  await page.getByLabel('Default model', { exact: true }).selectOption('demo-alternative');
  await expect(scheduleSave).toHaveCount(0);
  await modelSave.click();
  await expect(page.getByText('Model defaults saved for new executions.', { exact: true })).toBeVisible();
  await expect(modelSave).toHaveCount(0);
  await page.getByText('Models by role · all use default', { exact: true }).click();
  await page.getByLabel('Worker model', { exact: true }).selectOption('demo-default');
  await expect(modelSave).toBeEnabled();
  await page.getByRole('button', { name: 'Discard changes', exact: true }).click();
  const writes = await page.evaluate(() => window.__operationsRequests.filter(request => request.method !== 'GET'));
  if (JSON.stringify(writes.map(r => r.path)) !== JSON.stringify(['/v1/operations/settings/scheduler', '/v1/operations/settings/models'])) throw Error('Independent local settings writes drifted');
  await page.getByText('Add Git SSH key', { exact: true }).click();
  await expect(page.getByLabel('SSH private key', { exact: true })).toBeEnabled();
  await page.getByRole('radio', { name: 'Black', exact: true }).check();
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'black');
  await page.getByRole('radio', { name: 'System', exact: true }).check();
  controls.push('real independent settings PUTs using local transport, clean/dirty footers, bounds, role overrides, real Git editor, theme persistence');

  await open('settings', 390, 'black', 'attention');
  await up.click();
  await scheduleSave.click();
  await expect(page.getByRole('alert').filter({ hasText: 'Another operator saved Scheduler settings first.' })).toBeVisible();
  await expect(scheduleSave).toBeDisabled();
  await expect(modelLimit).toHaveValue('2');
  await page.getByRole('button', { name: 'Reset to saved values', exact: true }).click();
  await expect(modelLimit).toHaveValue('1');
  await expect(toolLimit).toHaveValue('3');
  await expect(scheduleSave).toHaveCount(0);
  await page.getByLabel('Default model', { exact: true }).selectOption('demo-alternative');
  await modelSave.click();
  await expect(modelSave).toBeDisabled();
  await expect(page.getByLabel('Default model', { exact: true })).toHaveValue('demo-alternative');
  await page.getByRole('button', { name: 'Use saved selection', exact: true }).click();
  await expect(modelSave).toHaveCount(0);
  controls.push('actual 412 conflict blocks resubmission, preserves model/scheduling drafts, resets independently');

  if (errors.length || networkViolations.length) throw Error(JSON.stringify({ errors, networkViolations }));
  await writeFile(new URL('review/metrics.json', root), JSON.stringify({ referenceFiles: Object.keys(manifest.files).length, layouts, controls, errors, networkViolations }, null, 2) + '\n');
  console.log(`Verified ${layouts.length} real UI layouts, ${controls.length} control groups and ${Object.keys(manifest.files).length} unchanged native reference files. No overflow, invalid API response, browser error or production/external API request.`);
} finally {
  await browser.close();
}
