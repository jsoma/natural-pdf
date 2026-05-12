# Vector: Region Overlap Memoized Arrays

Same rectangular overlap/exclusion fast path as `region_overlap_page_store`, but
memoizes derived arrays for identical element lists inside one run. It is not a
query-result cache; it tests whether production should have a versioned derived
array layer instead of rebuilding arrays on every region/exclusion call.
