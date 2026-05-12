# Vector: Spatial Char Filter Ephemeral

Fast-paths rectangular `filter_chars_spatially` calls with vector center and
overlap masks. Polygon targets and exclusions fall back to the production code.
