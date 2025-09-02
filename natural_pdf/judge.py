"""
Visual Judge for classifying regions based on image content.

This module provides a simple visual classifier that learns from examples
to classify regions (like checkboxes) into categories. It uses basic image
metrics rather than neural networks for fast, interpretable results.
"""

import hashlib
import json
import logging
import shutil
from collections import namedtuple
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Return types
Decision = namedtuple("Decision", ["label", "score"])
PickResult = namedtuple("PickResult", ["region", "index", "label", "score"])


class JudgeError(Exception):
    """Raised when Judge operations fail."""

    pass


class Judge:
    """
    Visual classifier for regions using simple image metrics.

    Requires class labels to be specified. For binary classification,
    requires at least one example of each class before making decisions.

    Examples:
        Checkbox detection:
        ```python
        judge = Judge("checkboxes", labels=["unchecked", "checked"])
        judge.add(empty_box, "unchecked")
        judge.add(marked_box, "checked")

        result = judge.decide(new_box)
        if result.label == "checked":
            print("Box is checked!")
        ```

        Signature detection:
        ```python
        judge = Judge("signatures", labels=["unsigned", "signed"])
        judge.add(blank_area, "unsigned")
        judge.add(signature_area, "signed")

        result = judge.decide(new_region)
        print(f"Classification: {result.label} (confidence: {result.score})")
        ```
    """

    def __init__(
        self,
        name: str,
        labels: List[str],
        base_dir: Optional[str] = None,
        target_prior: Optional[float] = None,
    ):
        """
        Initialize a Judge for visual classification.

        Args:
            name: Name for this judge (used for folder name)
            labels: Class labels (required, typically 2 for binary classification)
            base_dir: Base directory for storage. Defaults to current directory
            target_prior: Target prior probability for the FIRST label in the labels list.
                         - 0.5 (default) = neutral, treats both classes equally
                         - >0.5 = favors labels[0]
                         - <0.5 = favors labels[1]
                         Example: Judge("cb", ["checked", "unchecked"], target_prior=0.6)
                         favors detecting "checked" checkboxes.
        """
        if not labels or len(labels) != 2:
            raise JudgeError("Judge requires exactly 2 class labels (binary classification only)")

        self.name = name
        self.labels = labels
        self.target_prior = target_prior if target_prior is not None else 0.5

        # Set up directory structure
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.root_dir = self.base_dir / name
        self.root_dir.mkdir(exist_ok=True)

        # Create label directories
        for label in self.labels:
            (self.root_dir / label).mkdir(exist_ok=True)
        (self.root_dir / "unlabeled").mkdir(exist_ok=True)
        (self.root_dir / "_removed").mkdir(exist_ok=True)

        # Config file
        self.config_path = self.root_dir / "judge.json"

        # Load existing config or initialize
        self.thresholds = {}
        self.metrics_info = {}
        if self.config_path.exists():
            self._load_config()

    def add(self, region, label: Optional[str] = None) -> None:
        """
        Add a region to the judge's dataset.

        Args:
            region: Region object to add
            label: Class label. If None, added to unlabeled for later teaching

        Raises:
            JudgeError: If label is not in allowed labels
        """
        if label is not None and label not in self.labels:
            raise JudgeError(f"Label '{label}' not in allowed labels: {self.labels}")

        # Render region to image
        try:
            img = region.render(crop=True)
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
        except Exception as e:
            raise JudgeError(f"Failed to render region: {e}")

        # Convert to RGB if needed
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Generate hash from image content
        img_array = np.array(img)
        img_hash = hashlib.md5(img_array.tobytes()).hexdigest()[:12]

        # Determine target directory
        target_dir = self.root_dir / (label if label else "unlabeled")
        target_path = target_dir / f"{img_hash}.png"

        # Check if hash already exists anywhere
        existing_locations = []
        for check_label in self.labels + ["unlabeled", "_removed"]:
            check_path = self.root_dir / check_label / f"{img_hash}.png"
            if check_path.exists():
                existing_locations.append(check_label)

        if existing_locations:
            logger.warning(f"Duplicate image detected (hash: {img_hash})")
            logger.warning(f"Already exists in: {', '.join(existing_locations)}")
            print(f"⚠️  Duplicate image - already exists in: {', '.join(existing_locations)}")
            return

        # Save image
        img.save(target_path)
        logger.debug(f"Added image {img_hash} to {label if label else 'unlabeled'}")

    def teach(self, labels: Optional[List[str]] = None, review: bool = False) -> None:
        """
        Interactive teaching interface using IPython widgets.

        Args:
            labels: Labels to use for teaching. Defaults to self.labels
            review: If True, review already labeled images for re-classification
        """
        # Check for IPython environment
        try:
            import ipywidgets as widgets
            from IPython.display import clear_output, display
        except ImportError:
            raise JudgeError(
                "Teaching requires IPython and ipywidgets. Use 'pip install ipywidgets'"
            )

        labels = labels or self.labels

        # Get images to review
        if review:
            # Get all labeled images for review
            files_to_review = []
            for label in self.labels:
                label_dir = self.root_dir / label
                for img_path in sorted(label_dir.glob("*.png")):
                    files_to_review.append((img_path, label))

            if not files_to_review:
                print("No labeled images to review")
                return

            # Shuffle for review
            import random

            random.shuffle(files_to_review)
            review_files = [f[0] for f in files_to_review]
            original_labels = {str(f[0]): f[1] for f in files_to_review}
        else:
            # Get unlabeled images
            unlabeled_dir = self.root_dir / "unlabeled"
            review_files = sorted(unlabeled_dir.glob("*.png"))
            original_labels = {}

            if not review_files:
                print("No unlabeled images to teach")
                return

        # State for teaching
        self._teaching_state = {
            "current_index": 0,
            "labeled_count": 0,
            "removed_count": 0,
            "files": review_files,
            "labels": labels,
            "review_mode": review,
            "original_labels": original_labels,
        }

        # Create widgets
        image_widget = widgets.Image()
        status_label = widgets.Label()

        # Create buttons for labeling
        button_layout = widgets.Layout(width="auto", margin="5px")

        btn_prev = widgets.Button(description="↑ Previous", layout=button_layout)
        btn_class1 = widgets.Button(
            description=f"← {labels[0]}", layout=button_layout, button_style="primary"
        )
        btn_class2 = widgets.Button(
            description=f"→ {labels[1]}", layout=button_layout, button_style="success"
        )
        btn_skip = widgets.Button(description="↓ Skip", layout=button_layout)
        btn_remove = widgets.Button(
            description="✗ Remove", layout=button_layout, button_style="danger"
        )

        button_box = widgets.HBox([btn_prev, btn_class1, btn_class2, btn_skip, btn_remove])

        # Keyboard shortcuts info
        info_label = widgets.Label(
            value="Keys: ↑ prev | ← "
            + labels[0]
            + " | → "
            + labels[1]
            + " | ↓ skip | Delete remove"
        )

        def update_display():
            """Update the displayed image and status."""
            state = self._teaching_state
            if 0 <= state["current_index"] < len(state["files"]):
                img_path = state["files"][state["current_index"]]
                with open(img_path, "rb") as f:
                    image_widget.value = f.read()

                # Build status text
                status_text = f"Image {state['current_index'] + 1} of {len(state['files'])}"
                if state["review_mode"]:
                    current_label = state["original_labels"].get(str(img_path), "unknown")
                    status_text += f" (Currently: {current_label})"
                status_text += f" | Labeled: {state['labeled_count']}"
                if state["removed_count"] > 0:
                    status_text += f" | Removed: {state['removed_count']}"

                status_label.value = status_text

                # Update button states
                btn_prev.disabled = state["current_index"] == 0
            else:
                status_label.value = "Teaching complete!"
                # Hide the image widget instead of showing broken image
                image_widget.layout.display = "none"
                # Disable all buttons
                btn_prev.disabled = True
                btn_class1.disabled = True
                btn_class2.disabled = True
                btn_skip.disabled = True

                # Auto-retrain
                if state["labeled_count"] > 0 or state["removed_count"] > 0:
                    clear_output(wait=True)
                    print("Teaching complete!")
                    print(f"Labeled: {state['labeled_count']} images")
                    if state["removed_count"] > 0:
                        print(f"Removed: {state['removed_count']} images")

                    if state["labeled_count"] > 0:
                        print("\nRetraining with new examples...")
                        self._retrain()
                        print("✓ Training complete! Judge is ready to use.")
                else:
                    print("No changes made.")

        def move_file_to_class(class_index):
            """Move current file to specified class."""
            state = self._teaching_state
            if state["current_index"] >= len(state["files"]):
                return

            current_file = state["files"][state["current_index"]]
            target_dir = self.root_dir / labels[class_index]
            shutil.move(str(current_file), str(target_dir / current_file.name))
            state["labeled_count"] += 1
            state["current_index"] += 1
            update_display()

        # Button callbacks
        def on_prev(b):
            state = self._teaching_state
            if state["current_index"] > 0:
                state["current_index"] -= 1
                update_display()

        def on_class1(b):
            move_file_to_class(0)

        def on_class2(b):
            move_file_to_class(1)

        def on_skip(b):
            state = self._teaching_state
            state["current_index"] += 1
            update_display()

        def on_remove(b):
            state = self._teaching_state
            if state["current_index"] >= len(state["files"]):
                return

            current_file = state["files"][state["current_index"]]
            target_dir = self.root_dir / "_removed"
            shutil.move(str(current_file), str(target_dir / current_file.name))
            state["removed_count"] += 1
            state["current_index"] += 1
            update_display()

        # Connect buttons
        btn_prev.on_click(on_prev)
        btn_class1.on_click(on_class1)
        btn_class2.on_click(on_class2)
        btn_skip.on_click(on_skip)
        btn_remove.on_click(on_remove)

        # Create output widget for keyboard handling
        output = widgets.Output()

        # Keyboard event handler
        def on_key(event):
            """Handle keyboard events."""
            if event["type"] != "keydown":
                return

            key = event["key"]

            if key == "ArrowUp":
                on_prev(None)
            elif key == "ArrowLeft":
                on_class1(None)
            elif key == "ArrowRight":
                on_class2(None)
            elif key == "ArrowDown":
                on_skip(None)
            elif key in ["Delete", "Backspace"]:
                on_remove(None)

        # Display everything
        display(status_label)
        display(image_widget)
        display(button_box)
        display(info_label)
        display(output)

        # Show first image
        update_display()

        # Try to set up keyboard handling (may not work in all environments)
        try:
            from ipyevents import Event

            event_handler = Event(source=output, watched_events=["keydown"])
            event_handler.on_dom_event(on_key)
        except:
            # If ipyevents not available, just use buttons
            print("Note: Install ipyevents for keyboard shortcuts: pip install ipyevents")

    def decide(self, regions: Union["Region", List["Region"]]) -> Union[Decision, List[Decision]]:
        """
        Classify one or more regions.

        Args:
            regions: Single region or list of regions to classify

        Returns:
            Decision or list of Decisions with label and score

        Raises:
            JudgeError: If not enough training examples
        """
        # Check if we have examples
        for label in self.labels:
            label_dir = self.root_dir / label
            if not any(label_dir.glob("*.png")):
                raise JudgeError(f"Need at least one example of class '{label}' before deciding")

        # Ensure thresholds are current
        if not self.thresholds:
            self._retrain()

        # Handle single region
        single_input = not isinstance(regions, list)
        if single_input:
            regions = [regions]

        results = []
        for region in regions:
            # Extract metrics
            metrics = self._extract_metrics(region)

            # Apply thresholds with soft voting
            votes = {label: 0.0 for label in self.labels}
            total_weight = 0.0

            for metric_name, value in metrics.items():
                if metric_name in self.thresholds:
                    metric_info = self.thresholds[metric_name]
                    weight = metric_info["accuracy"]  # This is now Youden's J

                    # For binary classification
                    label1, label2 = self.labels
                    threshold1, direction1 = metric_info["thresholds"][label1]

                    # Get standard deviations for soft voting
                    stats = self.metrics_info.get(metric_name, {})
                    s1 = stats.get(f"std_{label1}", 0.0)
                    s2 = stats.get(f"std_{label2}", 0.0)
                    scale1 = s1 if s1 > 1e-6 else 1.0
                    scale2 = s2 if s2 > 1e-6 else 1.0

                    # Calculate signed margin (positive favors label1, negative favors label2)
                    if direction1 == "higher":
                        margin = (value - threshold1) / (scale1 if value >= threshold1 else scale2)
                    else:
                        margin = (threshold1 - value) / (scale1 if value <= threshold1 else scale2)

                    # Clip margin to avoid single metric dominating
                    margin = np.clip(margin, -6, 6)

                    # Soft votes using sigmoid
                    p1 = 1.0 / (1.0 + np.exp(-margin))
                    p2 = 1.0 - p1

                    votes[label1] += weight * p1
                    votes[label2] += weight * p2
                    total_weight += weight

            # Normalize votes
            if total_weight > 0:
                for label in votes:
                    votes[label] /= total_weight
            else:
                # Fallback: uniform votes so prior still works
                for label in votes:
                    votes[label] = 0.5
                total_weight = 1.0

            # Apply prior bias correction
            def _logit(p, eps=1e-6):
                p = max(eps, min(1 - eps, p))
                return np.log(p / (1 - p))

            def _sigmoid(x):
                if x >= 0:
                    z = np.exp(-x)
                    return 1.0 / (1.0 + z)
                else:
                    z = np.exp(x)
                    return z / (1.0 + z)

            # Estimate priors from training counts
            counts = self._get_training_counts()
            label1, label2 = self.labels
            n1 = counts.get(label1, 0)
            n2 = counts.get(label2, 0)
            total = max(1, n1 + n2)

            if n1 > 0 and n2 > 0:  # Only apply bias if we have examples of both classes
                emp_prior1 = n1 / total
                emp_prior2 = n2 / total

                # Target prior (0.5/0.5 neutralizes imbalance)
                target_prior1 = self.target_prior
                target_prior2 = 1.0 - self.target_prior

                # Calculate bias
                bias1 = _logit(target_prior1) - _logit(emp_prior1)
                bias2 = _logit(target_prior2) - _logit(emp_prior2)

                # Apply bias in logit space
                v1 = _sigmoid(_logit(votes[label1]) + bias1)
                v2 = _sigmoid(_logit(votes[label2]) + bias2)

                # Renormalize
                s = v1 + v2
                votes[label1] = v1 / s
                votes[label2] = v2 / s

            # Find best label
            best_label = max(votes.items(), key=lambda x: x[1])
            results.append(Decision(label=best_label[0], score=best_label[1]))

        return results[0] if single_input else results

    def pick(
        self, target_label: str, regions: List["Region"], labels: Optional[List[str]] = None
    ) -> PickResult:
        """
        Pick which region best matches the target label.

        Args:
            target_label: The class label to look for
            regions: List of regions to choose from
            labels: Optional human-friendly labels for each region

        Returns:
            PickResult with winning region, index, label (if provided), and score

        Raises:
            JudgeError: If target_label not in allowed labels
        """
        if target_label not in self.labels:
            raise JudgeError(f"Target label '{target_label}' not in allowed labels: {self.labels}")

        # Classify all regions
        decisions = self.decide(regions)

        # Find best match for target label
        best_index = -1
        best_score = -1.0

        for i, decision in enumerate(decisions):
            if decision.label == target_label and decision.score > best_score:
                best_score = decision.score
                best_index = i

        if best_index == -1:
            # No region matched the target label
            raise JudgeError(f"No region classified as '{target_label}'")

        # Build result
        region = regions[best_index]
        label = labels[best_index] if labels and best_index < len(labels) else None

        return PickResult(region=region, index=best_index, label=label, score=best_score)

    def count(self, target_label: str, regions: List["Region"]) -> int:
        """
        Count how many regions match the target label.

        Args:
            target_label: The class label to count
            regions: List of regions to check

        Returns:
            Number of regions classified as target_label
        """
        decisions = self.decide(regions)
        return sum(1 for d in decisions if d.label == target_label)

    def info(self) -> None:
        """
        Show configuration and training information for this Judge.
        """
        print(f"Judge: {self.name}")
        print(f"Labels: {self.labels}")
        if self.target_prior != 0.5:
            print(
                f"Target prior: {self.target_prior:.2f} (favors '{self.labels[0]}')"
                if self.target_prior > 0.5
                else f"Target prior: {self.target_prior:.2f} (favors '{self.labels[1]}')"
            )

        # Get training counts
        counts = self._get_training_counts()
        print(f"\nTraining examples:")
        for label in self.labels:
            count = counts.get(label, 0)
            print(f"  {label}: {count}")

        if counts.get("unlabeled", 0) > 0:
            print(f"  unlabeled: {counts['unlabeled']}")

        # Show actual imbalance
        labeled_counts = [counts.get(label, 0) for label in self.labels]
        if all(c > 0 for c in labeled_counts):
            max_count = max(labeled_counts)
            min_count = min(labeled_counts)
            if max_count != min_count:
                # Find which is which
                for i, label in enumerate(self.labels):
                    if counts.get(label, 0) == max_count:
                        majority_label = label
                    if counts.get(label, 0) == min_count:
                        minority_label = label

                ratio = max_count / min_count
                print(
                    f"\nClass imbalance: {majority_label}:{minority_label} = {max_count}:{min_count} ({ratio:.1f}:1)"
                )

                print("  Using Youden's J weights with soft voting and prior correction")

    def inspect(self, preview: bool = True) -> None:
        """
        Inspect all stored examples, showing their true labels and predicted labels/scores.
        Useful for debugging classification issues.

        Args:
            preview: If True (default), display images inline in HTML tables (requires IPython/Jupyter).
                     If False, use text-only output.
        """
        if not self.thresholds:
            print("No trained model yet. Add examples and the model will auto-train.")
            return

        if not preview:
            # Show basic info first
            self.info()
            print("-" * 80)

            print("\nThresholds learned:")
            for metric, info in self.thresholds.items():
                weight = info["accuracy"]  # This is now Youden's J
                selection_acc = info.get(
                    "selection_accuracy", info["accuracy"]
                )  # Fallback for old models
                print(f"  {metric}: weight={weight:.3f} (selection_accuracy={selection_acc:.3f})")
                for label, (threshold, direction) in info["thresholds"].items():
                    print(f"    {label}: {direction} than {threshold:.3f}")

                # Show metric distribution info if available
                if metric in self.metrics_info:
                    metric_stats = self.metrics_info[metric]
                    for label in self.labels:
                        mean_key = f"mean_{label}"
                        std_key = f"std_{label}"
                        if mean_key in metric_stats:
                            print(
                                f"    {label} distribution: mean={metric_stats[mean_key]:.3f}, std={metric_stats[std_key]:.3f}"
                            )

        if preview:
            # HTML preview mode
            try:
                import base64
                import io

                from IPython.display import HTML, display
            except ImportError:
                print("Preview mode requires IPython/Jupyter. Falling back to text mode.")
                preview = False

        if preview:
            # Build HTML tables for everything
            html_parts = []
            html_parts.append("<style>")
            html_parts.append("table { border-collapse: collapse; margin: 20px 0; }")
            html_parts.append("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }")
            html_parts.append("th { background-color: #f2f2f2; font-weight: bold; }")
            html_parts.append("img { max-width: 60px; max-height: 60px; }")
            html_parts.append(".correct { color: green; }")
            html_parts.append(".incorrect { color: red; }")
            html_parts.append(".metrics { font-size: 0.9em; color: #666; }")
            html_parts.append("h3 { margin-top: 30px; }")
            html_parts.append(".imbalance-warning { background-color: #fff3cd; color: #856404; }")
            html_parts.append("</style>")

            # Configuration table
            html_parts.append("<h3>Judge Configuration</h3>")
            html_parts.append("<table>")
            html_parts.append("<tr><th>Property</th><th>Value</th></tr>")
            html_parts.append(f"<tr><td>Name</td><td>{self.name}</td></tr>")
            html_parts.append(f"<tr><td>Labels</td><td>{', '.join(self.labels)}</td></tr>")
            html_parts.append(f"<tr><td>Target Prior</td><td>{self.target_prior:.2f}")
            if self.target_prior != 0.5:
                html_parts.append(
                    f" (favors '{self.labels[0] if self.target_prior > 0.5 else self.labels[1]}')"
                )
            html_parts.append("</td></tr>")
            html_parts.append("</table>")

            # Training counts table
            counts = self._get_training_counts()
            html_parts.append("<h3>Training Examples</h3>")
            html_parts.append("<table>")
            html_parts.append("<tr><th>Class</th><th>Count</th></tr>")

            # Check for imbalance
            labeled_counts = [counts.get(label, 0) for label in self.labels]
            is_imbalanced = False
            if all(c > 0 for c in labeled_counts):
                max_count = max(labeled_counts)
                min_count = min(labeled_counts)
                if max_count != min_count:
                    ratio = max_count / min_count
                    is_imbalanced = ratio > 1.5

            for label in self.labels:
                count = counts.get(label, 0)
                row_class = ""
                if is_imbalanced:
                    if count == max(labeled_counts):
                        row_class = ' class="imbalance-warning"'
                html_parts.append(f"<tr{row_class}><td>{label}</td><td>{count}</td></tr>")

            if counts.get("unlabeled", 0) > 0:
                html_parts.append(f"<tr><td>unlabeled</td><td>{counts['unlabeled']}</td></tr>")

            html_parts.append("</table>")

            if is_imbalanced:
                html_parts.append(
                    f"<p><em>Class imbalance detected ({ratio:.1f}:1). Using Youden's J weights with prior correction.</em></p>"
                )

            # Thresholds table
            html_parts.append("<h3>Learned Thresholds</h3>")
            html_parts.append("<table>")
            html_parts.append(
                "<tr><th>Metric</th><th>Weight (Youden's J)</th><th>Selection Accuracy</th><th>Threshold Details</th></tr>"
            )

            for metric, info in self.thresholds.items():
                weight = info["accuracy"]  # This is Youden's J
                selection_acc = info.get("selection_accuracy", weight)

                # Build threshold details
                details = []
                for label, (threshold, direction) in info["thresholds"].items():
                    details.append(f"<br>{label}: {direction} than {threshold:.3f}")

                # Add distribution info if available
                if metric in self.metrics_info:
                    metric_stats = self.metrics_info[metric]
                    details.append("<br><em>Distributions:</em>")
                    for label in self.labels:
                        mean_key = f"mean_{label}"
                        std_key = f"std_{label}"
                        if mean_key in metric_stats:
                            details.append(
                                f"<br>&nbsp;&nbsp;{label}: μ={metric_stats[mean_key]:.1f}, σ={metric_stats[std_key]:.1f}"
                            )

                html_parts.append("<tr>")
                html_parts.append(f"<td>{metric}</td>")
                html_parts.append(f"<td>{weight:.3f}</td>")
                html_parts.append(f"<td>{selection_acc:.3f}</td>")
                html_parts.append(f"<td>{''.join(details)}</td>")
                html_parts.append("</tr>")

            html_parts.append("</table>")

            all_correct = 0
            all_total = 0

            # First show labeled examples
            for true_label in self.labels:
                label_dir = self.root_dir / true_label
                examples = list(label_dir.glob("*.png"))

                if not examples:
                    continue

                html_parts.append(
                    f"<h3>Predictions: {true_label.upper()} ({len(examples)} total)</h3>"
                )
                html_parts.append("<table>")
                html_parts.append(
                    "<tr><th>Image</th><th>Status</th><th>Predicted</th><th>Score</th><th>Key Metrics</th></tr>"
                )

                correct = 0

                for img_path in sorted(examples)[:20]:  # Show max 20 per class in preview
                    # Load image
                    img = Image.open(img_path)
                    mock_region = type("MockRegion", (), {"render": lambda self, crop=True: img})()

                    # Get prediction
                    decision = self.decide(mock_region)
                    is_correct = decision.label == true_label
                    if is_correct:
                        correct += 1

                    # Extract metrics
                    metrics = self._extract_metrics(mock_region)

                    # Convert image to base64
                    buffered = io.BytesIO()
                    img.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode()

                    # Build row
                    status_class = "correct" if is_correct else "incorrect"
                    status_symbol = "✓" if is_correct else "✗"

                    # Format key metrics
                    metric_strs = []
                    for metric, value in sorted(metrics.items()):
                        if metric in self.thresholds:
                            metric_strs.append(f"{metric}={value:.1f}")
                    metrics_html = "<br>".join(metric_strs[:3])

                    html_parts.append("<tr>")
                    html_parts.append(f'<td><img src="data:image/png;base64,{img_str}" /></td>')
                    html_parts.append(f'<td class="{status_class}">{status_symbol}</td>')
                    html_parts.append(f"<td>{decision.label}</td>")
                    html_parts.append(f"<td>{decision.score:.3f}</td>")
                    html_parts.append(f'<td class="metrics">{metrics_html}</td>')
                    html_parts.append("</tr>")

                html_parts.append("</table>")

                accuracy = correct / len(examples) if examples else 0
                all_correct += correct
                all_total += len(examples)

                if len(examples) > 20:
                    html_parts.append(f"<p><em>... and {len(examples) - 20} more</em></p>")
                html_parts.append(
                    f"<p>Accuracy for {true_label}: <strong>{accuracy:.1%}</strong> ({correct}/{len(examples)})</p>"
                )

            if all_total > 0:
                overall_accuracy = all_correct / all_total
                html_parts.append(
                    f"<h3>Overall accuracy: {overall_accuracy:.1%} ({all_correct}/{all_total})</h3>"
                )

            # Now show unlabeled examples with predictions
            unlabeled_dir = self.root_dir / "unlabeled"
            unlabeled_examples = list(unlabeled_dir.glob("*.png"))

            if unlabeled_examples:
                html_parts.append(
                    f"<h3>Predictions: UNLABELED ({len(unlabeled_examples)} total)</h3>"
                )
                html_parts.append("<table>")
                html_parts.append(
                    "<tr><th>Image</th><th>Predicted</th><th>Score</th><th>Key Metrics</th></tr>"
                )

                for img_path in sorted(unlabeled_examples)[:20]:  # Show max 20
                    # Load image
                    img = Image.open(img_path)
                    mock_region = type("MockRegion", (), {"render": lambda self, crop=True: img})()

                    # Get prediction
                    decision = self.decide(mock_region)

                    # Extract metrics
                    metrics = self._extract_metrics(mock_region)

                    # Convert image to base64
                    buffered = io.BytesIO()
                    img.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode()

                    # Format key metrics
                    metric_strs = []
                    for metric, value in sorted(metrics.items()):
                        if metric in self.thresholds:
                            metric_strs.append(f"{metric}={value:.1f}")
                    metrics_html = "<br>".join(metric_strs[:3])

                    html_parts.append("<tr>")
                    html_parts.append(f'<td><img src="data:image/png;base64,{img_str}" /></td>')
                    html_parts.append(f"<td>{decision.label}</td>")
                    html_parts.append(f"<td>{decision.score:.3f}</td>")
                    html_parts.append(f'<td class="metrics">{metrics_html}</td>')
                    html_parts.append("</tr>")

                html_parts.append("</table>")

                if len(unlabeled_examples) > 20:
                    html_parts.append(
                        f"<p><em>... and {len(unlabeled_examples) - 20} more</em></p>"
                    )

            # Display HTML
            display(HTML("".join(html_parts)))

        else:
            # Text mode (original)
            print("\nPredictions on training data:")
            print("-" * 80)

            # Test each labeled example
            all_correct = 0
            all_total = 0

            for true_label in self.labels:
                label_dir = self.root_dir / true_label
                examples = list(label_dir.glob("*.png"))

                if not examples:
                    continue

                print(f"\n{true_label.upper()} examples ({len(examples)} total):")
                correct = 0

                for img_path in sorted(examples)[:10]:  # Show max 10 per class
                    # Load image and create mock region
                    img = Image.open(img_path)
                    mock_region = type("MockRegion", (), {"render": lambda self, crop=True: img})()

                    # Get prediction
                    decision = self.decide(mock_region)
                    is_correct = decision.label == true_label
                    if is_correct:
                        correct += 1

                    # Extract metrics for this example
                    metrics = self._extract_metrics(mock_region)

                    # Show result
                    status = "✓" if is_correct else "✗"
                    print(
                        f"  {status} {img_path.name}: predicted={decision.label} (score={decision.score:.3f})"
                    )

                    # Show key metric values
                    metric_strs = []
                    for metric, value in sorted(metrics.items()):
                        if metric in self.thresholds:
                            metric_strs.append(f"{metric}={value:.2f}")
                    if metric_strs:
                        print(f"     Metrics: {', '.join(metric_strs[:3])}")

                accuracy = correct / len(examples) if examples else 0
                all_correct += correct
                all_total += len(examples)

                if len(examples) > 10:
                    print(f"  ... and {len(examples) - 10} more")
                print(f"  Accuracy for {true_label}: {accuracy:.1%} ({correct}/{len(examples)})")

            if all_total > 0:
                overall_accuracy = all_correct / all_total
                print(f"\nOverall accuracy: {overall_accuracy:.1%} ({all_correct}/{all_total})")

            # Show unlabeled examples with predictions
            unlabeled_dir = self.root_dir / "unlabeled"
            unlabeled_examples = list(unlabeled_dir.glob("*.png"))

            if unlabeled_examples:
                print(f"\nUNLABELED examples ({len(unlabeled_examples)} total) - predictions:")

                for img_path in sorted(unlabeled_examples)[:10]:  # Show max 10
                    # Load image and create mock region
                    img = Image.open(img_path)
                    mock_region = type("MockRegion", (), {"render": lambda self, crop=True: img})()

                    # Get prediction
                    decision = self.decide(mock_region)

                    # Extract metrics
                    metrics = self._extract_metrics(mock_region)

                    print(
                        f"  {img_path.name}: predicted={decision.label} (score={decision.score:.3f})"
                    )

                    # Show key metric values
                    metric_strs = []
                    for metric, value in sorted(metrics.items()):
                        if metric in self.thresholds:
                            metric_strs.append(f"{metric}={value:.2f}")
                    if metric_strs:
                        print(f"     Metrics: {', '.join(metric_strs[:3])}")

                if len(unlabeled_examples) > 10:
                    print(f"  ... and {len(unlabeled_examples) - 10} more")

    def lookup(self, region) -> Optional[Tuple[str, Image.Image]]:
        """
        Look up a region and return its hash and image if found in training data.

        Args:
            region: Region to look up

        Returns:
            Tuple of (hash, image) if found, None if not found
        """
        try:
            # Generate hash for the region
            img = region.render(crop=True)
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            if img.mode != "RGB":
                img = img.convert("RGB")
            img_array = np.array(img)
            img_hash = hashlib.md5(img_array.tobytes()).hexdigest()[:12]

            # Look for the image in all directories
            for subdir in ["checked", "unchecked", "unlabeled", "_removed"]:
                if subdir == "checked" or subdir == "unchecked":
                    # Only look in valid label directories
                    if subdir not in self.labels:
                        continue

                img_path = self.root_dir / subdir / f"{img_hash}.png"
                if img_path.exists():
                    stored_img = Image.open(img_path)
                    logger.debug(f"Found region in '{subdir}' with hash {img_hash}")
                    return (img_hash, stored_img)

            logger.debug(f"Region not found in training data (hash: {img_hash})")
            return None

        except Exception as e:
            logger.error(f"Failed to lookup region: {e}")
            return None

    def show(self, max_per_class: int = 10, size: Tuple[int, int] = (100, 100)) -> None:
        """
        Display a grid showing examples from each category.

        Args:
            max_per_class: Maximum number of examples to show per class
            size: Size of each image in pixels (width, height)
        """
        try:
            import ipywidgets as widgets
            from IPython.display import display
            from PIL import Image as PILImage
        except ImportError:
            print("Show requires IPython and ipywidgets")
            return

        # Collect images from each category
        categories = {}
        total_counts = {}
        for label in self.labels:
            label_dir = self.root_dir / label
            all_images = list(label_dir.glob("*.png"))
            total_counts[label] = len(all_images)
            images = sorted(all_images)[:max_per_class]
            if images:
                categories[label] = images

        # Add unlabeled if any
        unlabeled_dir = self.root_dir / "unlabeled"
        all_unlabeled = list(unlabeled_dir.glob("*.png"))
        total_counts["unlabeled"] = len(all_unlabeled)
        unlabeled = sorted(all_unlabeled)[:max_per_class]
        if unlabeled:
            categories["unlabeled"] = unlabeled

        if not categories:
            print("No images to show")
            return

        # Create grid layout
        rows = []

        # Check for class imbalance
        labeled_counts = {k: v for k, v in total_counts.items() if k != "unlabeled"}
        if labeled_counts and len(labeled_counts) >= 2:
            max_count = max(labeled_counts.values())
            min_count = min(labeled_counts.values())
            if min_count > 0 and max_count / min_count > 3:
                warning = widgets.HTML(
                    f'<div style="background: #fff3cd; padding: 10px; margin: 10px 0; border: 1px solid #ffeeba; border-radius: 4px;">'
                    f"<strong>⚠️ Class imbalance detected:</strong> {labeled_counts}<br>"
                    f"Consider adding more examples of the minority class for better accuracy."
                    f"</div>"
                )
                rows.append(warning)

        for category, image_paths in categories.items():
            # Category header showing total count
            shown = len(image_paths)
            total = total_counts[category]
            header_text = f"<h3>{category}"
            if shown < total:
                header_text += f" ({shown} of {total} shown)"
            else:
                header_text += f" ({total} total)"
            header_text += "</h3>"
            header = widgets.HTML(header_text)

            # Image row
            image_widgets = []
            for img_path in image_paths:
                # Load and resize image
                img = PILImage.open(img_path)
                img.thumbnail(size, PILImage.Resampling.LANCZOS)

                # Convert to bytes for display
                import io

                img_bytes = io.BytesIO()
                img.save(img_bytes, format="PNG")
                img_bytes.seek(0)

                # Create image widget
                img_widget = widgets.Image(value=img_bytes.read(), width=size[0], height=size[1])
                image_widgets.append(img_widget)

            # Create horizontal box for this category
            category_box = widgets.VBox([header, widgets.HBox(image_widgets)])
            rows.append(category_box)

        # Display all categories
        display(widgets.VBox(rows))

    def forget(self, region: Optional["Region"] = None, delete: bool = False) -> None:
        """
        Clear training data, delete all files, or move a specific region to unlabeled.

        Args:
            region: If provided, move this specific region to unlabeled
            delete: If True, permanently delete all files
        """
        # Handle specific region case
        if region is not None:
            # Get hash of the region
            try:
                img = region.render(crop=True)
                if not isinstance(img, Image.Image):
                    img = Image.fromarray(img)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img_array = np.array(img)
                img_hash = hashlib.md5(img_array.tobytes()).hexdigest()[:12]
            except Exception as e:
                logger.error(f"Failed to hash region: {e}")
                return

            # Find and move the image
            moved = False
            for label in self.labels + ["_removed"]:
                source_path = self.root_dir / label / f"{img_hash}.png"
                if source_path.exists():
                    target_path = self.root_dir / "unlabeled" / f"{img_hash}.png"
                    shutil.move(str(source_path), str(target_path))
                    print(f"Moved region from '{label}' to 'unlabeled'")
                    moved = True
                    break

            if not moved:
                print(f"Region not found in training data")
            return

        # Handle delete or clear training
        if delete:
            # Delete entire directory
            if self.root_dir.exists():
                shutil.rmtree(self.root_dir)
                print(f"Deleted all data for judge '{self.name}'")
            else:
                print(f"No data found for judge '{self.name}'")

            # Reset internal state
            self.thresholds = {}
            self.metrics_info = {}

            # Recreate directory structure
            self.root_dir.mkdir(exist_ok=True)
            for label in self.labels:
                (self.root_dir / label).mkdir(exist_ok=True)
            (self.root_dir / "unlabeled").mkdir(exist_ok=True)
            (self.root_dir / "_removed").mkdir(exist_ok=True)

        else:
            # Just clear training (move everything to unlabeled)
            moved_count = 0

            # Move all labeled images back to unlabeled
            unlabeled_dir = self.root_dir / "unlabeled"
            for label in self.labels:
                label_dir = self.root_dir / label
                if label_dir.exists():
                    for img_path in label_dir.glob("*.png"):
                        shutil.move(str(img_path), str(unlabeled_dir / img_path.name))
                        moved_count += 1

            # Clear thresholds
            self.thresholds = {}
            self.metrics_info = {}

            # Remove saved config
            if self.config_path.exists():
                self.config_path.unlink()

            print(f"Moved {moved_count} labeled images back to unlabeled.")
            print("Training data cleared. Judge is now untrained.")

    def save(self, path: Optional[str] = None) -> None:
        """
        Save the judge configuration (auto-retrains first).

        Args:
            path: Optional path to save to. Defaults to judge.json in root directory
        """
        # Retrain with current examples
        self._retrain()

        # Save config
        save_path = Path(path) if path else self.config_path

        config = {
            "name": self.name,
            "labels": self.labels,
            "target_prior": self.target_prior,
            "thresholds": self.thresholds,
            "metrics_info": self.metrics_info,
            "training_counts": self._get_training_counts(),
        }

        with open(save_path, "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Saved judge to {save_path}")

    @classmethod
    def load(cls, path: str) -> "Judge":
        """
        Load a judge from a saved configuration.

        Args:
            path: Path to the saved judge.json file or the judge directory

        Returns:
            Loaded Judge instance
        """
        path = Path(path)

        # If path is a directory, look for judge.json inside
        if path.is_dir():
            config_path = path / "judge.json"
            base_dir = path.parent
            name = path.name
        else:
            config_path = path
            base_dir = path.parent.parent if path.parent.name != "." else path.parent
            # Try to infer name from path
            name = None

        with open(config_path, "r") as f:
            config = json.load(f)

        # Use saved name if we couldn't infer it
        if name is None:
            name = config["name"]

        # Create judge with saved config
        judge = cls(
            name,
            labels=config["labels"],
            base_dir=base_dir,
            target_prior=config.get("target_prior", 0.5),
        )  # Default to 0.5 for old configs
        judge.thresholds = config["thresholds"]
        judge.metrics_info = config.get("metrics_info", {})

        return judge

    # Private methods

    def _extract_metrics(self, region) -> Dict[str, float]:
        """Extract image metrics from a region."""
        try:
            img = region.render(crop=True)
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)

            # Convert to grayscale for analysis
            gray = np.array(img.convert("L"))

            metrics = {}

            # 1. Center darkness
            h, w = gray.shape
            cy, cx = h // 2, w // 2
            center_size = min(5, h // 4, w // 4)  # Adaptive center size
            center = gray[
                max(0, cy - center_size) : min(h, cy + center_size + 1),
                max(0, cx - center_size) : min(w, cx + center_size + 1),
            ]
            metrics["center_darkness"] = 255 - np.mean(center)

            # 2. Overall darkness (ink density)
            metrics["ink_density"] = 255 - np.mean(gray)

            # 3. Dark pixel ratio
            metrics["dark_pixel_ratio"] = np.sum(gray < 200) / gray.size

            # 4. Standard deviation (complexity)
            metrics["std_dev"] = np.std(gray)

            # 5. Edge vs center ratio
            edge_size = max(2, min(h // 10, w // 10))
            edge_mask = np.zeros_like(gray, dtype=bool)
            edge_mask[:edge_size, :] = True
            edge_mask[-edge_size:, :] = True
            edge_mask[:, :edge_size] = True
            edge_mask[:, -edge_size:] = True

            edge_mean = np.mean(gray[edge_mask]) if np.any(edge_mask) else 255
            center_mean = np.mean(center)
            metrics["edge_center_ratio"] = edge_mean / (center_mean + 1)

            # 6. Diagonal density (for X patterns)
            if h > 10 and w > 10:
                diag_mask = np.zeros_like(gray, dtype=bool)
                for i in range(min(h, w)):
                    if i < h and i < w:
                        diag_mask[i, i] = True
                        diag_mask[i, w - 1 - i] = True
                metrics["diagonal_density"] = 255 - np.mean(gray[diag_mask])
            else:
                metrics["diagonal_density"] = metrics["ink_density"]

            return metrics

        except Exception as e:
            raise JudgeError(f"Failed to extract metrics: {e}")

    def _retrain(self) -> None:
        """Retrain thresholds from current examples."""
        # Collect all examples
        examples = {label: [] for label in self.labels}

        for label in self.labels:
            label_dir = self.root_dir / label
            for img_path in label_dir.glob("*.png"):
                img = Image.open(img_path)
                # Create a mock region that just returns the image
                mock_region = type("MockRegion", (), {"render": lambda self, crop=True: img})()
                metrics = self._extract_metrics(mock_region)
                examples[label].append(metrics)

        # Check we have examples
        for label, exs in examples.items():
            if not exs:
                logger.warning(f"No examples for class '{label}'")
                return

        # Check for class imbalance
        example_counts = {label: len(exs) for label, exs in examples.items()}
        max_count = max(example_counts.values())
        min_count = min(example_counts.values())

        imbalance_ratio = max_count / min_count if min_count > 0 else float("inf")
        is_imbalanced = imbalance_ratio > 1.5  # Consider imbalanced if more than 1.5x difference

        if is_imbalanced:
            logger.info(
                f"Class imbalance detected: {example_counts} (ratio {imbalance_ratio:.1f}:1)"
            )
            logger.info("Using balanced accuracy for threshold selection")

        # Find best thresholds for each metric
        self.thresholds = {}
        self.metrics_info = {}
        metric_candidates = []  # Store all metrics with their scores

        all_metrics = set()
        for exs in examples.values():
            for ex in exs:
                all_metrics.update(ex.keys())

        for metric in all_metrics:
            # Get all values for this metric
            values_by_label = {}
            for label, exs in examples.items():
                values_by_label[label] = [ex.get(metric, 0) for ex in exs]

            # Find threshold that best separates classes (for binary)
            if len(self.labels) == 2:
                label1, label2 = self.labels
                vals1 = values_by_label[label1]
                vals2 = values_by_label[label2]

                # Try different thresholds
                all_vals = vals1 + vals2
                best_threshold = None
                best_accuracy = 0
                best_direction = None

                for threshold in np.percentile(all_vals, [10, 20, 30, 40, 50, 60, 70, 80, 90]):
                    # Test both directions
                    for direction in ["higher", "lower"]:
                        if direction == "higher":
                            correct1 = sum(1 for v in vals1 if v > threshold)
                            correct2 = sum(1 for v in vals2 if v <= threshold)
                        else:
                            correct1 = sum(1 for v in vals1 if v < threshold)
                            correct2 = sum(1 for v in vals2 if v >= threshold)

                        # Always use balanced accuracy for threshold selection
                        # This finds fair thresholds regardless of class imbalance
                        acc1 = correct1 / len(vals1) if len(vals1) > 0 else 0
                        acc2 = correct2 / len(vals2) if len(vals2) > 0 else 0
                        accuracy = (acc1 + acc2) / 2

                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_threshold = threshold
                            best_direction = direction

                # Calculate Youden's J statistic for weight (TPR - FPR)
                if best_direction == "higher":
                    tp = sum(1 for v in vals1 if v > best_threshold)
                    fn = len(vals1) - tp
                    tn = sum(1 for v in vals2 if v <= best_threshold)
                    fp = len(vals2) - tn
                else:
                    tp = sum(1 for v in vals1 if v < best_threshold)
                    fn = len(vals1) - tp
                    tn = sum(1 for v in vals2 if v >= best_threshold)
                    fp = len(vals2) - tn

                tpr = tp / len(vals1) if len(vals1) > 0 else 0
                fpr = fp / len(vals2) if len(vals2) > 0 else 0
                youden_j = max(0.0, min(1.0, tpr - fpr))

                # Store all candidates
                metric_candidates.append(
                    {
                        "metric": metric,
                        "youden_j": youden_j,
                        "selection_accuracy": best_accuracy,
                        "threshold": best_threshold,
                        "direction": best_direction,
                        "label1": label1,
                        "label2": label2,
                        "stats": {
                            "mean_" + label1: np.mean(vals1),
                            "mean_" + label2: np.mean(vals2),
                            "std_" + label1: np.std(vals1),
                            "std_" + label2: np.std(vals2),
                        },
                    }
                )

        # Sort by selection accuracy
        metric_candidates.sort(key=lambda x: x["selection_accuracy"], reverse=True)

        # Use relaxed cutoff when imbalanced
        keep_cutoff = 0.55 if is_imbalanced else 0.60

        # Keep metrics that pass cutoff, or top 3 if none pass
        kept_metrics = [m for m in metric_candidates if m["selection_accuracy"] > keep_cutoff]
        if not kept_metrics and metric_candidates:
            # Keep top 3 metrics even if they don't pass cutoff
            kept_metrics = metric_candidates[:3]
            logger.warning(
                f"No metrics passed cutoff {keep_cutoff}, keeping top {len(kept_metrics)} metrics"
            )

        # Store selected metrics
        for candidate in kept_metrics:
            metric = candidate["metric"]
            label1 = candidate["label1"]
            label2 = candidate["label2"]
            self.thresholds[metric] = {
                "accuracy": candidate["youden_j"],  # Use Youden's J as weight
                "selection_accuracy": candidate["selection_accuracy"],
                "thresholds": {
                    label1: (candidate["threshold"], candidate["direction"]),
                    label2: (
                        candidate["threshold"],
                        "lower" if candidate["direction"] == "higher" else "higher",
                    ),
                },
            }
            self.metrics_info[metric] = candidate["stats"]

    def _load_config(self) -> None:
        """Load configuration from file."""
        try:
            with open(self.config_path, "r") as f:
                config = json.load(f)

            self.thresholds = config.get("thresholds", {})
            self.metrics_info = config.get("metrics_info", {})

            # Verify labels match
            if config.get("labels") != self.labels:
                logger.warning(
                    f"Saved labels {config.get('labels')} don't match current {self.labels}"
                )

        except Exception as e:
            logger.warning(f"Failed to load config: {e}")

    def _get_training_counts(self) -> Dict[str, int]:
        """Get count of examples per class."""
        counts = {}
        for label in self.labels:
            label_dir = self.root_dir / label
            counts[label] = len(list(label_dir.glob("*.png")))
        counts["unlabeled"] = len(list((self.root_dir / "unlabeled").glob("*.png")))
        return counts
