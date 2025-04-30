import subprocess
import tarfile
import zipfile
from pathlib import Path

DIST_DIR = Path("dist")


def build_package():
    subprocess.run(["python", "-m", "build", "--sdist", "--wheel"], check=True)


def get_sdist_files():
    sdist_path = next(DIST_DIR.glob("*.tar.gz"))
    with tarfile.open(sdist_path, "r:gz") as tar:
        return sorted(str(Path(m.name)) for m in tar.getmembers() if m.isfile())


def get_wheel_files():
    wheel_path = next(DIST_DIR.glob("*.whl"))
    with zipfile.ZipFile(wheel_path, "r") as zipf:
        return sorted(str(f) for f in zipf.namelist() if not f.endswith("/"))


def get_gitignored_files():
    proc = subprocess.run(
        ["git", "ls-files", "--others", "-i", "--exclude-standard"],
        check=True,
        capture_output=True,
        text=True,
    )
    return sorted(proc.stdout.strip().splitlines())


def diff_lists(packaged, ignored):
    return sorted(set(packaged) & set(ignored))


def main():
    build_package()

    sdist_files = get_sdist_files()
    wheel_files = get_wheel_files()
    ignored_files = get_gitignored_files()

    print("\n🚫 Files in *sdist* that are also .gitignored:")
    for f in diff_lists(sdist_files, ignored_files):
        print("  •", f)

    print("\n🚫 Files in *wheel* that are also .gitignored:")
    for f in diff_lists(wheel_files, ignored_files):
        print("  •", f)


if __name__ == "__main__":
    main()
