"""Mixin to add visual template matching to Page/PDF/PDFCollection"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple, Union

from tqdm.auto import tqdm

if TYPE_CHECKING:
    from natural_pdf.elements.base import Element
    from natural_pdf.elements.region import Region
    from natural_pdf.vision.results import MatchResults


class VisualSearchMixin:
    """Add visual template matching helpers to classes that include this mixin"""

    def match_template(
        self,
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

        Args:
            examples: Single element/region or list of examples whose appearance should be matched.
            confidence: Minimum similarity score (0-1).
            sizes: Size variations to search. Accepts a float (±percentage), tuple range, tuple with
                explicit step, or a list of exact multipliers.
            resolution: DPI used to render both templates and target pages/regions.
            hash_size: Perceptual hash grid size (only used when ``method="phash"``).
            step: Explicit sliding window step in pixels (defaults to 10% of template size).
            method: Matching algorithm – ``"phash"`` (default) or ``"template"`` for direct correlation.
            max_per_page: Optional cap on matches returned per page/region.
            show_progress: Whether to emit a tqdm progress bar while scanning.
            mask_threshold: Optional threshold (0-1) to treat bright pixels as background in the
                rendered template; helpful for logos/stamps on white paper.

        Returns:
            MatchResults collection
        """
        from natural_pdf.vision.results import MatchResults  # Local import to avoid circular

        warnings.warn(
            "natural-pdf's visual template matching remains experimental and currently relies on "
            "perceptual hash / template matching only; behaviour may change in future releases.",
            UserWarning,
            stacklevel=2,
        )

        # Ensure examples is a list
        if isinstance(examples, (tuple, list)):
            example_list: Sequence[Union["Element", "Region"]] = list(examples)
        else:
            example_list = [examples]

        from .similarity import VisualMatcher, compute_phash

        # Initialize matcher with specified hash size
        matcher = VisualMatcher(hash_size=hash_size)

        # Prepare templates
        templates = []
        # Extract mask_threshold from kwargs for phash
        mask_threshold_255 = (
            int(mask_threshold * 255) if mask_threshold is not None and method == "phash" else None
        )

        for example in example_list:
            # Render the example region/element
            example_image = example.render(resolution=resolution, crop=True)
            template_hash = compute_phash(
                example_image, hash_size=hash_size, mask_threshold=mask_threshold_255
            )
            templates.append({"image": example_image, "hash": template_hash, "source": example})

        # Get pages to search based on the object type
        if hasattr(self, "__class__") and self.__class__.__name__ == "PDFCollection":
            # PDFCollection needs to iterate through all PDFs
            pages_to_search = []
            for pdf in self:
                pages_to_search.extend(pdf.pages)
        elif hasattr(self, "pages"):  # PDF
            pages_to_search = self.pages
        elif hasattr(self, "number"):  # Single page
            pages_to_search = [self]
        elif hasattr(self, "page") and hasattr(self, "bbox"):  # Region
            pages_to_search = [self]
        else:
            raise TypeError(f"Cannot search in {type(self)}")

        # Calculate total operations for progress bar
        total_operations = 0
        if show_progress:
            # Get scales that will be searched
            scales = matcher._get_search_scales(sizes)

            # Pre-calculate for all pages and templates
            for search_obj in pages_to_search:
                # Estimate image size based on object type
                if hasattr(search_obj, "page") and hasattr(search_obj, "bbox"):
                    # Region
                    page_w = int(search_obj.width * resolution / 72.0)
                    page_h = int(search_obj.height * resolution / 72.0)
                else:
                    # Page
                    page_w = int(search_obj.width * resolution / 72.0)
                    page_h = int(search_obj.height * resolution / 72.0)

                for template_data in templates:
                    template_w, template_h = template_data["image"].size

                    for scale in scales:
                        scaled_w = int(template_w * scale)
                        scaled_h = int(template_h * scale)

                        if scaled_w <= page_w and scaled_h <= page_h:
                            # Determine step size
                            if step is not None:
                                actual_step = step
                            else:
                                # Default to 10% of template size
                                actual_step = max(1, int(min(scaled_w, scaled_h) * 0.1))

                            x_windows = len(range(0, page_w - scaled_w + 1, actual_step))
                            y_windows = len(range(0, page_h - scaled_h + 1, actual_step))
                            total_operations += x_windows * y_windows

        # Search each page
        all_matches = []

        # Create single progress bar for all operations
        progress_bar = None
        operations_done = 0
        last_update = 0
        update_frequency = max(1, total_operations // 1000)  # Update at most 1000 times

        if show_progress and total_operations > 0:
            progress_bar = tqdm(
                total=total_operations,
                desc="Searching",
                unit="window",
                miniters=update_frequency,  # Minimum iterations between updates
                mininterval=0.1,  # Minimum time between updates (seconds)
            )

        for page_idx, search_obj in enumerate(pages_to_search):
            # Determine if we're searching in a page or a region
            if hasattr(search_obj, "page") and hasattr(search_obj, "bbox"):
                # This is a Region - render only the region area
                region = search_obj
                page = region.page
                page_image = region.render(resolution=resolution, crop=True)
                # Region offset for coordinate conversion
                region_x0, region_y0 = region.x0, region.top
            else:
                # This is a Page - render the full page
                page = search_obj
                page_image = page.render(resolution=resolution)
                region_x0, region_y0 = 0, 0

            # Convert page coordinates to image coordinates
            scale = resolution / 72.0  # PDF is 72 DPI

            page_matches = []

            # Search for each template
            for template_idx, template_data in enumerate(templates):
                template_image = template_data["image"]
                template_hash = template_data["hash"]

                # Custom progress callback to update our main progress bar
                def update_progress():
                    nonlocal operations_done, last_update
                    operations_done += 1

                    # Only update progress bar every N operations to avoid overwhelming output
                    if progress_bar and (
                        operations_done - last_update >= update_frequency
                        or operations_done == total_operations
                    ):
                        progress_bar.update(operations_done - last_update)
                        last_update = operations_done

                        # Update description with current page/template info
                        if len(pages_to_search) > 1:
                            progress_bar.set_description(
                                f"Page {page.number}/{len(pages_to_search)}"
                            )
                        elif len(templates) > 1:
                            progress_bar.set_description(
                                f"Template {template_idx + 1}/{len(templates)}"
                            )

                # Find matches in this page - never show internal progress
                candidates = matcher.find_matches_in_image(
                    template_image,
                    page_image,
                    template_hash=template_hash,
                    confidence_threshold=confidence,
                    sizes=sizes,
                    step=step,
                    method=method,
                    show_progress=False,  # We handle progress ourselves
                    progress_callback=update_progress if progress_bar else None,
                    mask_threshold=mask_threshold,
                )

                # Convert image coordinates back to PDF coordinates
                for candidate in candidates:
                    img_x0, img_y0, img_x1, img_y1 = candidate.bbox

                    # Convert from image pixels to PDF points
                    # No flipping needed! PDF coordinates map directly to PIL coordinates
                    pdf_x0 = img_x0 / scale + region_x0
                    pdf_y0 = img_y0 / scale + region_y0
                    pdf_x1 = img_x1 / scale + region_x0
                    pdf_y1 = img_y1 / scale + region_y0

                    from .results import Match

                    # Create Match object
                    match = Match(
                        page=page,
                        bbox=(pdf_x0, pdf_y0, pdf_x1, pdf_y1),
                        confidence=candidate.confidence,
                        source_example=template_data["source"],
                    )
                    page_matches.append(match)

            # Apply max_per_page limit if specified
            if max_per_page and len(page_matches) > max_per_page:
                # Sort by confidence and take top N
                page_matches.sort(key=lambda m: m.confidence, reverse=True)
                page_matches = page_matches[:max_per_page]

            all_matches.extend(page_matches)

        # Close progress bar
        if progress_bar:
            progress_bar.close()

        return MatchResults(all_matches)

    def find_similar(
        self,
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
    ) -> "MatchResults":
        """
        Backwards-compatible wrapper for the old visual search entry point.

        This method is deprecated; use :meth:`match_template` instead.
        """

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
