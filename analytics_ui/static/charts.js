// Thin Chart.js helpers themed for the explorer UI. Exposes window.CX.
// Loaded after vendor/chart.umd.js and before app.js. If Chart.js failed to
// load, CX.ready is false and callers fall back to text.
(function () {
  const CX = (window.CX = window.CX || {});
  CX.ready = typeof window.Chart !== 'undefined';
  if (!CX.ready) return;

  Chart.defaults.color = '#8b93a7';
  Chart.defaults.font.family = 'ui-monospace, SFMono-Regular, Menlo, monospace';
  Chart.defaults.font.size = 11;
  Chart.defaults.borderColor = 'rgba(255,255,255,0.07)';
  Chart.defaults.maintainAspectRatio = false;
  Chart.defaults.animation = false;
  Chart.defaults.plugins.legend.labels.boxWidth = 10;
  Chart.defaults.plugins.legend.labels.boxHeight = 10;

  const PALETTE = ['#5fc8f7', '#34d399', '#fbbf24', '#a78bfa', '#f87171',
                   '#f472b6', '#60a5fa', '#facc15', '#2dd4bf', '#fb923c'];
  CX.PALETTE = PALETTE;

  const GRID = { color: 'rgba(255,255,255,0.05)' };

  // Build a fixed-height wrapper + canvas and instantiate the chart once it's
  // laid out (Chart.js needs a non-zero parent height).
  CX.chart = function (config, height) {
    const wrap = document.createElement('div');
    wrap.className = 'chart-wrap';
    wrap.style.height = (height || 240) + 'px';
    const canvas = document.createElement('canvas');
    wrap.appendChild(canvas);
    requestAnimationFrame(() => {
      try { new Chart(canvas, config); } catch (e) { /* leave blank on failure */ }
    });
    return wrap;
  };

  // Vertical bar (optionally stacked when datasets share a `stack` key).
  CX.bars = function (labels, datasets, opts) {
    opts = opts || {};
    const ds = datasets.map((d, i) => ({
      label: d.label, data: d.data, stack: d.stack,
      backgroundColor: d.color || PALETTE[i % PALETTE.length],
      borderWidth: 0, borderRadius: 2, maxBarThickness: 46,
    }));
    return {
      type: 'bar',
      data: { labels, datasets: ds },
      options: {
        indexAxis: opts.horizontal ? 'y' : 'x',
        scales: {
          x: { stacked: !!opts.stacked, grid: GRID, ticks: { autoSkip: false } },
          y: { stacked: !!opts.stacked, grid: GRID, beginAtZero: true },
        },
        plugins: { legend: { display: datasets.length > 1 } },
      },
    };
  };
})();
