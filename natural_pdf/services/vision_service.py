from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Iterable, List, Optional, Sequence, Tuple, Union, cast

from tqdm.auto import tqdm

from natural_pdf.core.interfaces import (
    HasRenderablePages,
    RenderablePage,
    RenderableRegion,
    SupportsPDFCollection,
)
from natural_pdf.services.registry import register_delegate

SearchTarget = Union[RenderablePage, RenderableRegion]


class VisualSearchService:
    """Service powering visual template matching helpers."""

    def __init__(self, context):
        self._context = context

    @register_delegate("vision", "match_template")
    def match_template(
        self,
        host,
        examples: Union["Element", "Region", List[Union["Element", "Region"]]],
        confidence: float = 0.6,
        sizes: Optional[Union[float, Tuple, List]] = (0.8, 1.2),
        resolution: int = 72,
        hash_size: int = 20,
        step: Optional[int] = None,
        method: str = "phash",
        max_per_page: Optional[int] = None,
        show_progress: bool = True,
        mask_threshold: Optional[float] = None,
    ) -> "MatchResults":
        """
        Match rendered templates against the rendered page/region.
        """
        from natural_pdf.vision.results import Match, MatchResults
        from natural_pdf.vision.similarity import VisualMatcher, compute_phash

        warnings.warn(
            "natural-pdf's visual template matching remains experimental and currently relies on "
            "perceptual hash / template matching only; behaviour may change in future releases.",
            UserWarning,
            stacklevel=2,
        )

        if isinstance(examples, (tuple, list)):
            example_list: Sequence[Union["Element", "Region"]] = list(examples)
        else:
            example_list = [examples]

        matcher = VisualMatcher(hash_size=hash_size)

        templates = []
        mask_threshold_255 = (
            int(mask_threshold * 255) if mask_threshold is not None and method == "phash" else None
        )

        for example in example_list:
            example_image = example.render(resolution=resolution, crop=True)
            if example_image is None:
                raise ValueError("Unable to render template example to an image.")
            template_hash = compute_phash(
                example_image,
                hash_size=hash_size,
                mask_threshold=mask_threshold_255,
            )
            templates.append({"image": example_image, "hash": template_hash, "source": example})

        pages_to_search = self._collect_search_targets(host)

        total_operations = 0
        progress_bar = None
        operations_done = 0
        last_update = 0

        if show_progress:
            scales = matcher._get_search_scales(sizes)
            for search_obj in pages_to_search:
                page_w = int(search_obj.width * resolution / 72.0)
                page_h = int(search_obj.height * resolution / 72.0)
                for template_data in templates:
                    template_w, template_h = template_data["image"].size
                    for scale in scales:
                        scaled_w = int(template_w * scale)
                        scaled_h = int(template_h * scale)
                        if scaled_w <= page_w and scaled_h <= page_h:
                            if step is not None:
                                actual_step = step
                            else:
                                actual_step = max(1, int(min(scaled_w, scaled_h) * 0.1))
                            x_windows = len(range(0, page_w - scaled_w + 1, actual_step))
                            y_windows = len(range(0, page_h - scaled_h + 1, actual_step))
                            total_operations += x_windows * y_windows

            if total_operations > 0:
                update_frequency = max(1, total_operations // 1000)
                progress_bar = tqdm(
                    total=total_operations,
                    desc="Searching",
                    unit="window",
                    miniters=update_frequency,
                    mininterval=0.1,
                )

        all_matches = []
        for idx, search_obj in enumerate(pages_to_search):
            if isinstance(search_obj, RenderableRegion):
                region = search_obj
                page = cast(RenderablePage, region.page)
                page_image = region.render(resolution=resolution, crop=True)
                if page_image is None:
                    raise ValueError("Region.render returned None during visual search.")
                region_x0, region_y0 = region.x0, region.top
            else:
                page = cast(RenderablePage, search_obj)
                page_image = page.render(resolution=resolution)
                if page_image is None:
                    raise ValueError("Page.render returned None during visual search.")
                region_x0, region_y0 = 0, 0

            scale = resolution / 72.0

            page_matches = []

            for template_idx, template_data in enumerate(templates):
                template_image = template_data["image"]
                template_hash = template_data["hash"]

                def update_progress():
                    nonlocal operations_done, last_update
                    operations_done += 1
                    if progress_bar and (
                        operations_done - last_update >= max(1, total_operations // 1000)
                        or operations_done == total_operations
                    ):
                        progress_bar.update(operations_done - last_update)
                        last_update = operations_done
                        if len(pages_to_search) > 1:
                            progress_bar.set_description(f"Page {idx + 1}/{len(pages_to_search)}")
                        elif len(templates) > 1:
                            progress_bar.set_description(
                                f"Template {template_idx + 1}/{len(templates)}"
                            )

                candidates = matcher.find_matches_in_image(
                    template_image,
                    page_image,
                    template_hash=template_hash,
                    confidence_threshold=confidence,
                    sizes=sizes,
                    step=step,
                    method=method,
                    show_progress=False,
                    progress_callback=update_progress if progress_bar else None,
                    mask_threshold=mask_threshold,
                )

                for candidate in candidates:
                    img_x0, img_y0, img_x1, img_y1 = candidate.bbox
                    pdf_x0 = img_x0 / scale + region_x0
                    pdf_y0 = img_y0 / scale + region_y0
                    pdf_x1 = img_x1 / scale + region_x0
                    pdf_y1 = img_y1 / scale + region_y0

                    match = Match(
                        page=page,
                        bbox=(pdf_x0, pdf_y0, pdf_x1, pdf_y1),
                        confidence=candidate.confidence,
                        source_example=template_data["source"],
                    )
                    page_matches.append(match)

            if max_per_page and len(page_matches) > max_per_page:
                page_matches.sort(key=lambda m: m.confidence, reverse=True)
                page_matches = page_matches[:max_per_page]
            all_matches.extend(page_matches)

        if progress_bar:
            progress_bar.close()

        return MatchResults(all_matches)

    @register_delegate("vision", "find_similar")
    def find_similar(
        self,
        host,
        examples: Union["Element", "Region", List[Union["Element", "Region"]]],
        using: str = "vision",
        confidence: float = 0.6,
        sizes: Optional[Union[float, Tuple, List]] = (0.8, 1.2),
        resolution: int = 72,
        hash_size: int = 20,
        step: Optional[int] = None,
        method: str = "phash",
        max_per_page: Optional[int] = None,
        show_progress: bool = True,
        **kwargs,
    ):
        warnings.warn(
            "VisualSearchMixin.find_similar() is deprecated and will be removed in a future "
            "release. Use match_template(...) instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        if using != "vision":
            raise NotImplementedError(f"using='{using}' is no longer supported.")

        mask_threshold = kwargs.pop("mask_threshold", None)
        if kwargs:
            unexpected = ", ".join(sorted(kwargs.keys()))
            raise TypeError(f"find_similar() got unexpected keyword arguments: {unexpected}")

        return self.match_template(
            host,
            examples=examples,
            confidence=confidence,
            sizes=sizes,
            resolution=resolution,
            hash_size=hash_size,
            step=step,
            method=method,
            max_per_page=max_per_page,
            show_progress=show_progress,
            mask_threshold=mask_threshold,
        )

    def _collect_search_targets(self, host) -> List[SearchTarget]:
        if isinstance(host, SupportsPDFCollection):
            targets: List[SearchTarget] = []
            for pdf in host:
                if not isinstance(pdf, HasRenderablePages):
                    raise TypeError(
                        "Objects yielded by PDF collections must expose a 'pages' sequence."
                    )
                targets.extend(list(cast(Sequence[RenderablePage], pdf.pages)))
            return targets

        if isinstance(host, HasRenderablePages):
            return list(cast(Sequence[RenderablePage], host.pages))

        if isinstance(host, RenderableRegion):
            return [host]

        if isinstance(host, RenderablePage):
            return [host]

        # Flow objects should delegate to their analysis region
        analysis_region = getattr(host, "_analysis_region", None)
        if callable(analysis_region):
            region = analysis_region()
            if region is not None:
                return self._collect_search_targets(region)

        raise TypeError(f"Cannot perform visual search in {type(host)}")


if TYPE_CHECKING:  # pragma: no cover
    from natural_pdf.elements.base import Element
    from natural_pdf.elements.region import Region
