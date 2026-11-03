# Contractor Settings refinement

The user approves the other Operations tabs and asks for a more polished Settings.
The reference is deliberately focused: exact Settings DOM, existing theme tokens,
shared icon markup and functional base CSS. No shell or controller reading needed.

Please make this calm and refined, with a clear visual hierarchy rather than a
stack of interchangeable forms. Preserve the two-column idea: execution models
and scheduling on the wider left, smaller repository/appearance controls on the
right. Keep the personal column useful without stretching its cards to the whole
row. Use deliberate spacing, compact headings, subtle icon treatments, clear
selected theme swatches and well aligned steppers. Make the default model feel
like the main decision; role overrides and diagnostics are progressive disclosures.

All data and behavior already exist. Do not add, hide or replace text or controls.
There are independent dirty/save/conflict states for model and scheduling forms;
footers appear only while editing. [hidden] must keep working. The error fixture
can hide Server forms; personal theme controls remain usable. Git private key input
is deliberately disabled: it is only a layout example, not credential generation.
The base template has system/light/dark/black choices; all share project fonts.

Output ONLY new opendesign.css, about 120–200 lines. Scope every rule to
.settings-redesign (or its descendants); media queries can wrap scoped rules.
Use only tokens already declared in the reference for colors, type and spacing.
No imports, assets, new fonts or JS/HTML edits. At 1100px stack execution and a
pair of personal cards; at 390/320px use one column, readable controls, no clipped
steppers or labels. Keep desktop rhythm and avoid excessive mobile blank space.
No dependency installation, browser QA, network requests or Skill discovery.
The host assembles exact CSS into the full-shell seed and verifies all behavior.
