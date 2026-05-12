# Vector: Decorations Ephemeral

Builds structure-of-arrays views inside each decoration call and uses vector
masks over chars for strike, underline, and highlight detection.

This tests the cold single-use cost of vectorization without relying on derived
array reuse.
