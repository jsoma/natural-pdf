# find_contains_first

Hypothesis: common one-off calls like `page.find("text:contains(Date)")` should
not need to allocate an `ElementCollection` of every matching word and then take
the first item.

This patch short-circuits `SelectorService.find()` for simple, non-regex
`text:contains(...)` selectors on pages and rectangular regions. It falls back
for exclusions, custom selector engines, text tolerance overrides, and complex
selector syntax.

This is a cold-path structural shortcut, not a cache.
