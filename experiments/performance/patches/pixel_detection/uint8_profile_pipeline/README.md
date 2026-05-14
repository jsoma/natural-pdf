# Pixel Detection: UInt8 Profile Pipeline

Hypothesis: line projection detection spends avoidable time converting the
binarized image to float and copying it per axis. Keep the binary image as
`uint8`/bool and compute projection profiles with `count_nonzero`.

Risk: low to medium. The output should be close to current behavior, but integer
grayscale is not bit-for-bit identical to the float luminance dot product.
