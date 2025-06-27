class QAResult(dict):
    """Dictionary-like container for Document QA results with a convenient ``show`` method.

    This class behaves exactly like a regular ``dict`` so existing code that
    expects a mapping will continue to work.  In addition it exposes:

    • ``show()`` – delegates to the underlying ``source_elements.show`` if those
      elements are present (added automatically by ``ask_pdf_page`` and
      ``ask_pdf_region``).  This provides a quick way to visualise where an
      answer was found in the document.

    • Attribute access (e.g. ``result.answer``) as sugar for the usual
      ``result["answer"]``.
    """

    # ---------------------------------------------------------------------
    # Convenience helpers
    # ---------------------------------------------------------------------
    def show(self, *args, **kwargs):
        """Display the answer region by delegating to ``source_elements.show``.

        Any positional or keyword arguments are forwarded to
        ``ElementCollection.show``.
        """
        source = self.get("source_elements")
        if source is None:
            raise AttributeError("QAResult does not contain 'source_elements'; nothing to show().")
        if not hasattr(source, "show"):
            raise AttributeError("'source_elements' object has no 'show' method; cannot visualise.")
        return source.show(*args, **kwargs)

    # ------------------------------------------------------------------
    # Attribute <-> key delegation so ``result.answer`` works
    # ------------------------------------------------------------------
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def __setattr__(self, key, value):
        # Store all non-dunder attributes in the underlying mapping so that
        # they remain serialisable.
        if key.startswith("__") and key.endswith("__"):
            super().__setattr__(key, value)
        else:
            self[key] = value

    # Ensure ``copy`` keeps the subclass type
    def copy(self):
        return QAResult(self)
