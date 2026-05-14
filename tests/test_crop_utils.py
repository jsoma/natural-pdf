from natural_pdf.core.crop_utils import resolve_crop_bbox


def test_resolve_crop_bbox_does_not_compute_content_for_uncropped_render():
    calls = 0

    def content_bbox():
        nonlocal calls
        calls += 1
        return (10, 20, 30, 40)

    assert (
        resolve_crop_bbox(
            width=100,
            height=100,
            crop=False,
            content_bbox_fn=content_bbox,
        )
        is None
    )
    assert calls == 0


def test_resolve_crop_bbox_does_not_compute_content_for_explicit_bbox():
    calls = 0

    def content_bbox():
        nonlocal calls
        calls += 1
        return (10, 20, 30, 40)

    assert resolve_crop_bbox(
        width=100,
        height=100,
        crop_bbox=(1, 2, 3, 4),
        content_bbox_fn=content_bbox,
    ) == (1, 2, 3, 4)
    assert calls == 0


def test_resolve_crop_bbox_computes_content_for_crop_true():
    calls = 0

    def content_bbox():
        nonlocal calls
        calls += 1
        return (10, 20, 30, 40)

    assert resolve_crop_bbox(
        width=100,
        height=100,
        crop=True,
        content_bbox_fn=content_bbox,
    ) == (10, 20, 30, 40)
    assert calls == 1
