# Concepts

The mental model behind the API — why things work the way they do, and the
honest edges. Read these when a method surprises you or when you want to
predict behavior instead of trial-and-erroring it.

- **[The spatial model](spatial-model.md)** — everything is boxes; how
  directional navigation, anchors, `until=`, and `within=` actually resolve.
- **[How text becomes elements](text-and-elements.md)** — chars → words →
  elements, tolerance settings, and why `text:contains()` can miss a phrase
  that `extract_text()` shows.
- **[Exclusions](exclusions.md)** — read-time views, not deletions; when
  they bind, what honors them, when not to use them.
- **[Selectors](selectors.md)** — the grammar, how it differs from CSS, and
  the region-type vocabulary that layout engines produce.
- **[Tables, a decision guide](tables.md)** — the recovery ladder from
  bordered tables down to scanned ones.
- **[Engines and models](engines-and-models.md)** — what downloads what and
  how big, local vs remote VLMs, and the privacy contract for
  `set_default_client()`.
