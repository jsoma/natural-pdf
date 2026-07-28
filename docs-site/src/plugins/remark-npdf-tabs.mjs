/**
 * remark-npdf-tabs
 *
 * The docs staging script (scripts/docs_stage.py) converts authored
 * `/// tab | Label` groups into HTML-comment markers in the page body:
 *
 *   <!-- npdf-tabs:start -->
 *   <!-- npdf-tab:start label="pdfplumber" -->
 *   (arbitrary markdown: paragraphs, fenced code, raw HTML output)
 *   <!-- npdf-tab:end -->
 *   <!-- npdf-tab:start label="Natural PDF" -->
 *   ...
 *   <!-- npdf-tab:end -->
 *   <!-- npdf-tabs:end -->
 *
 * This plugin finds those marker nodes in the mdast tree and wraps the
 * enclosed content so the rendered HTML is:
 *
 *   <div class="npdf-tabs" data-npdf-tabs>
 *     <section class="npdf-tab" data-label="pdfplumber">
 *       <p class="npdf-tab-label">pdfplumber</p>
 *       ...rendered markdown content...
 *     </section>
 *     ...
 *   </div>
 *
 * The original mdast nodes between the markers are kept as-is (markdown
 * inside a tab is still processed normally); only raw-HTML open/close nodes
 * are inserted around them. Astro's markdown pipeline (rehype-raw) re-nests
 * the interleaved fragments into a proper DOM tree.
 *
 * With JavaScript disabled the output degrades to labeled stacked sections;
 * src/scripts/tabs.ts progressively enhances each group into a tablist.
 *
 * Unbalanced or malformed markers throw, with the page path in the message.
 */

const TABS_START = /^<!--\s*npdf-tabs:start\s*-->$/;
const TABS_END = /^<!--\s*npdf-tabs:end\s*-->$/;
const TAB_START = /^<!--\s*npdf-tab:start\s+label="([^"]*)"\s*-->$/;
const TAB_END = /^<!--\s*npdf-tab:end\s*-->$/;
const ANY_MARKER = /<!--\s*npdf-tabs?:(?:start|end)(?:\s+label="[^"]*")?\s*-->/g;

function escapeAttr(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll('"', "&quot;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function raw(value) {
  return { type: "html", value };
}

/**
 * CommonMark can merge a marker comment into an adjacent raw-HTML block when
 * no blank line separates them (e.g. `<div>...</div>` directly followed by
 * `<!-- npdf-tab:end -->`). Split any html node whose value mixes marker
 * comments with other content into separate nodes so markers always stand
 * alone.
 */
function splitMarkerNodes(children) {
  const out = [];
  for (const node of children) {
    if (node.type !== "html" || !ANY_MARKER.test(node.value)) {
      ANY_MARKER.lastIndex = 0;
      out.push(node);
      continue;
    }
    ANY_MARKER.lastIndex = 0;
    let last = 0;
    const pieces = [];
    for (const match of node.value.matchAll(ANY_MARKER)) {
      const before = node.value.slice(last, match.index);
      if (before.trim() !== "") pieces.push(before.trim());
      pieces.push(match[0]);
      last = match.index + match[0].length;
    }
    const after = node.value.slice(last);
    if (after.trim() !== "") pieces.push(after.trim());
    for (const piece of pieces) {
      out.push({ type: "html", value: piece });
    }
  }
  return out;
}

function isMarker(node, pattern) {
  return node.type === "html" && pattern.test(node.value.trim());
}

function transformParent(parent, where) {
  if (!parent.children) return;

  // Recurse first so markers nested inside e.g. blockquotes are handled too.
  for (const child of parent.children) {
    transformParent(child, where);
  }

  if (
    !parent.children.some(
      (node) =>
        node.type === "html" && /npdf-tabs?:/.test(node.value)
    )
  ) {
    return;
  }

  const children = splitMarkerNodes(parent.children);
  const out = [];
  let i = 0;

  const fail = (message) => {
    throw new Error(`remark-npdf-tabs: ${message} in ${where}`);
  };

  while (i < children.length) {
    const node = children[i];

    if (isMarker(node, TAB_START) || isMarker(node, TAB_END)) {
      fail("npdf-tab marker outside an npdf-tabs block");
    }
    if (isMarker(node, TABS_END)) {
      fail("npdf-tabs:end without a matching npdf-tabs:start");
    }
    if (!isMarker(node, TABS_START)) {
      out.push(node);
      i += 1;
      continue;
    }

    // Inside an npdf-tabs block.
    i += 1;
    out.push(raw('<div class="npdf-tabs" data-npdf-tabs>'));
    let closed = false;
    let tabCount = 0;

    while (i < children.length) {
      const inner = children[i];

      if (isMarker(inner, TABS_END)) {
        i += 1;
        closed = true;
        break;
      }

      const startMatch =
        inner.type === "html" && inner.value.trim().match(TAB_START);
      if (!startMatch) {
        if (isMarker(inner, TABS_START)) {
          fail("nested npdf-tabs:start");
        }
        fail("content between tabs that is not inside an npdf-tab block");
      }

      const label = startMatch[1];
      i += 1;
      out.push(
        raw(`<section class="npdf-tab" data-label="${escapeAttr(label)}">`)
      );
      out.push(raw(`<p class="npdf-tab-label">${escapeAttr(label)}</p>`));

      let tabClosed = false;
      while (i < children.length) {
        const tabNode = children[i];
        if (isMarker(tabNode, TAB_END)) {
          i += 1;
          tabClosed = true;
          break;
        }
        if (
          isMarker(tabNode, TAB_START) ||
          isMarker(tabNode, TABS_START) ||
          isMarker(tabNode, TABS_END)
        ) {
          fail(`npdf-tab:start label="${label}" without a matching npdf-tab:end`);
        }
        out.push(tabNode);
        i += 1;
      }
      if (!tabClosed) {
        fail(`npdf-tab:start label="${label}" without a matching npdf-tab:end`);
      }
      out.push(raw("</section>"));
      tabCount += 1;
    }

    if (!closed) {
      fail("npdf-tabs:start without a matching npdf-tabs:end");
    }
    if (tabCount === 0) {
      fail("npdf-tabs block with no npdf-tab entries");
    }
    out.push(raw("</div>"));
  }

  parent.children = out;
}

export default function remarkNpdfTabs() {
  return function transformer(tree, file) {
    const where =
      (file && (file.path || (file.history && file.history[0]))) ||
      "unknown file";
    transformParent(tree, where);
  };
}
