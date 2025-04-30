import os
import sys

# Default to standard tqdm
try:
    from tqdm.std import tqdm as selected_tqdm
except ImportError:
    # Basic fallback if even std is missing (though unlikely)
    def selected_tqdm(*args, **kwargs):
        iterable = args[0] if args else None
        if iterable:
            return iterable
        return None  # Simple passthrough if no iterable


# Try to detect notebook environment
try:
    # Check 1: Are we running in an IPython kernel?
    from IPython import get_ipython

    ipython = get_ipython()
    if ipython and "IPKernelApp" in ipython.config:
        # Check 2: Is it likely a notebook UI (Jupyter Notebook/Lab, VSCode, etc.)?
        # This checks for common indicators. Might not be foolproof.
        if "VSCODE_PID" in os.environ or (
            "ipykernel" in sys.modules and "spyder" not in sys.modules
        ):
            # Check 3: Can we import notebook version?
            try:
                from tqdm.notebook import tqdm as notebook_tqdm

                selected_tqdm = notebook_tqdm  # Use notebook version
            except ImportError:
                pass  # Stick with std if notebook version missing
except ImportError:
    pass  # Stick with std if IPython not available


def get_tqdm():
    """Returns the tqdm class best suited for the detected environment."""
    return selected_tqdm


# Example usage (for testing):
if __name__ == "__main__":
    import time

    tqdm_instance = get_tqdm()
    print(f"Using tqdm class: {tqdm_instance}")
    for i in tqdm_instance(range(10), desc="Testing tqdm"):
        time.sleep(0.1)
