# Pixel Detection: Fast Top-K Peaks

Hypothesis: guide workflows usually cap lines with `max_lines_h/v`; selecting
top local maxima directly from projection profiles may avoid scipy
peak/prominence overhead.

Risk: medium to high. This is a comparator for the peak-selection phase and may
choose different lines on noisy pages.
