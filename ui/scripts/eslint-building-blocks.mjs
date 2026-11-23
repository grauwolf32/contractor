// ESLint rule "building-blocks/use-shared" (V385-001 A8): outside src/ui and
// src/app, pages use the shared building blocks instead of writing their own
//
//   - a <button> with the ui-btn class   -> Button, IconButton (src/ui)
//   - role="alertdialog"                 -> AlertDialog (src/ui)
//   - an inline <svg> icon               -> Icon (src/app/icon.tsx)
//
// Files that still contain them are listed in building-blocks-baseline.json
// with their current counts. A file may not exceed its count, and a count
// that is higher than the file needs fails too, so the baseline only
// shrinks: lower it in the change that adopts the building block. Drawings
// that are not icons (charts, the Studio canvas, the sign-in backdrop) are
// listed under "drawings" with a reason and may keep their <svg>.

import { readFileSync } from "node:fs";
import { relative } from "node:path";

const KINDS = {
  button:
    "a <button> with the ui-btn class; use Button or IconButton from src/ui",
  alertdialog: 'role="alertdialog"; use AlertDialog from src/ui',
  svg: "an inline <svg>; add the icon to src/app/icon.tsx and use <Icon>",
};

/** Reads the baseline file: { files: { path: { kind: count } }, drawings: { path: reason } }. */
export function readBaseline(path) {
  return JSON.parse(readFileSync(path, "utf8"));
}

function hasUiBtnClass(attribute) {
  const value = attribute.value;
  if (value === null) return false;
  const texts = [];
  const collect = (node) => {
    if (node === null || typeof node !== "object") return;
    if (node.type === "Literal" && typeof node.value === "string")
      texts.push(node.value);
    if (node.type === "TemplateElement") texts.push(node.value.cooked ?? "");
    for (const key of Object.keys(node)) {
      if (key === "parent") continue;
      const child = node[key];
      if (Array.isArray(child)) child.forEach(collect);
      else if (child && typeof child.type === "string") collect(child);
    }
  };
  collect(value);
  return texts.some((text) => /(^|\s)ui-btn(\s|$)/.test(text));
}

const rule = {
  meta: {
    type: "suggestion",
    docs: { description: "Use the shared Button, AlertDialog and Icon" },
    schema: [
      {
        type: "object",
        properties: {
          root: { type: "string" },
          baseline: { type: "object" },
        },
        additionalProperties: false,
      },
    ],
    messages: {
      found:
        "{{what}} ({{count}} in this file, {{allowed}} allowed by scripts/building-blocks-baseline.json).",
      stale:
        "scripts/building-blocks-baseline.json allows {{allowed}} of {{kind}} here but the file has {{count}}; lower the baseline to {{count}}.",
    },
  },
  create(context) {
    const options = context.options[0] ?? {};
    const root = options.root ?? context.cwd;
    const baseline = options.baseline ?? { files: {}, drawings: {} };
    const file = relative(root, context.filename).split("\\").join("/");
    const allowedFor = (kind) =>
      kind === "svg" && Object.hasOwn(baseline.drawings ?? {}, file)
        ? Infinity
        : (baseline.files?.[file]?.[kind] ?? 0);
    const found = { button: [], alertdialog: [], svg: [] };
    return {
      JSXOpeningElement(node) {
        if (node.name.type !== "JSXIdentifier") return;
        if (node.name.name === "svg") found.svg.push(node);
        if (node.name.name === "button") {
          const className = node.attributes.find(
            (attribute) =>
              attribute.type === "JSXAttribute" &&
              attribute.name.name === "className",
          );
          if (className !== undefined && hasUiBtnClass(className))
            found.button.push(node);
        }
        for (const attribute of node.attributes) {
          if (
            attribute.type === "JSXAttribute" &&
            attribute.name.name === "role" &&
            attribute.value?.type === "Literal" &&
            attribute.value.value === "alertdialog"
          )
            found.alertdialog.push(attribute);
        }
      },
      "Program:exit"(program) {
        for (const kind of Object.keys(KINDS)) {
          const nodes = found[kind];
          const allowed = allowedFor(kind);
          if (nodes.length > allowed) {
            for (const node of nodes.slice(allowed))
              context.report({
                node,
                messageId: "found",
                data: {
                  what: KINDS[kind],
                  count: String(nodes.length),
                  allowed: String(allowed),
                },
              });
          } else if (allowed !== Infinity && nodes.length < allowed) {
            context.report({
              node: program,
              messageId: "stale",
              data: {
                kind,
                count: String(nodes.length),
                allowed: String(allowed),
              },
            });
          }
        }
      },
    };
  },
};

export default {
  meta: { name: "building-blocks" },
  rules: { "use-shared": rule },
};
