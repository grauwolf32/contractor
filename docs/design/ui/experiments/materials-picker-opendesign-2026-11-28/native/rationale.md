# Rationale

An additive layer for the Materials workspace and the New check material chooser. It loads after layout.css and every color, spacing, radius and font value comes from the existing theme tokens. The global font sizes, the 76px navigation rail and the Project tabs are untouched.

1. Kind-tinted leading icons. Source rows take the selection blue, API rows the info tokens, Engagement rows the success tokens and diff rows the warning tokens. The filename stays the primary signal. The tint only speeds up scanning within a group and across groups, and the icon shapes still differ.

2. One row system for both journeys. Library rows and chooser rows share the same padding, name line and two muted meta lines. Format, size and update time sit on the first line. The Git commit and short description sit on the second. Revisions render as quiet code chips so an exact revision is easy to find again in a list, in the preview header and in the dialog footer.

3. Column balance in the library. The preview column becomes a bounded 320 to 384px beside the flexible list with a 20px gutter. The list keeps the wider share, long filenames wrap with overflow-wrap instead of truncating, and the preview still fits code blocks and endpoint paths.

4. Reader hierarchy. The kicker becomes a small uppercase label, the title steps up to 16px, and the namespace path moves to the footer in faint monospace. The full path is context rather than a competitor of the name. The details arrow becomes a small bordered control under the content divider, and code blocks get the code background and border so a source material reads as code.

5. Workbench order and labels. The objective stays the first element of the page and the type list is sticky beside it on desktop. Slot labels become uppercase micro labels with the Required or Optional tag pushed to the right edge of the row. Attachments get even padding, a 13px name and the same kind tint as the library. Summary facts align as label and value rows so Check type and Materials read at a glance.

6. Chooser focus. The preview half keeps its sunken background so the dialog reads as list and candidate. The selected row highlights its check action. The footer names the chosen file with its exact revision chip. Attach material stays the only primary button and Cancel keeps the earlier attachment intact. Add material remains available inside the chooser.

7. Responsive composition. The breakpoints stay at 1200px and 820px with one tighter step at 480px. Because this layer loads last, every value the scaffold redefines at a breakpoint is restated here. That keeps the narrower preview column and the two column check layout working at the right widths instead of letting the wider desktop values win everywhere.

8. Motion. One 150ms transition covers background, border and color on rows, chips, type cards and buttons, and it is disabled under prefers-reduced-motion. No other animation is added.

Mobile treatment. From 820px down the chooser dialog becomes a flex column. The header and footer shrink to their own content and the material list takes the remaining height, so the chrome leaves room for actual materials instead of a large floating region. The list shows first and selecting a row opens the preview with the existing Back to materials action, which preserves the filtered list on return. The preview body drops its fixed 300px minimum height so it hugs its content. At 480px rows and dialog padding tighten further and the footer buttons reach a 44px touch target.
