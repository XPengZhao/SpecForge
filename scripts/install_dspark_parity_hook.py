"""Install/remove the opt-in DSpark parity hook in a specified vLLM checkout."""
import argparse
import ast
from pathlib import Path
import shutil

MARKER = "# DSPARK_PARITY_HOOK_V1"
IMPORT = "\nfrom .parity_debug import enabled as _parity_enabled, propose_with_dump\n"
METHOD = '''
    # DSPARK_PARITY_HOOK_V1
    def propose(self, *args, **kwargs):
        return propose_with_dump(self, super().propose, args, kwargs)

'''


def instrument(source):
    if MARKER in source:
        return source
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DSparkSpeculator")
    if any(isinstance(n, ast.FunctionDef) and n.name == "propose" for n in cls.body):
        raise ValueError("Existing DSparkSpeculator.propose override; inspect before installing")
    sample = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_sample_sequential")
    lines = source.splitlines(keepends=True)
    # Preserve a method docstring if this vLLM revision has one.
    first = sample.body[0]
    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
        insertion = first.end_lineno
    else:
        insertion = first.lineno - 1
    lines.insert(insertion, "        if _parity_enabled():\n            self._dspark_parity_hidden = head_hidden\n")
    # Lower edits first keep class/module line offsets valid.
    lines.insert(cls.body[0].lineno - 1, METHOD)
    lines.insert(cls.lineno - 1, IMPORT)
    result = "".join(lines)
    ast.parse(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-root", type=Path, required=True)
    parser.add_argument("--restore", action="store_true")
    args = parser.parse_args()
    target = args.vllm_root / "vllm/v1/worker/gpu/spec_decode/dspark/speculator.py"
    backup = target.with_suffix(".py.parity-backup")
    helper = target.with_name("parity_debug.py")
    if args.restore:
        if not backup.exists():
            raise ValueError("No parity backup found")
        # Refuse to discard changes made after installation.
        if target.read_text() != instrument(backup.read_text()):
            raise ValueError("vLLM file changed after installation; restore manually from backup")
        target.write_text(backup.read_text())
        backup.unlink()
        helper.unlink(missing_ok=True)
        print("Restored", target)
        return
    original = target.read_text()
    if MARKER in original:
        if not backup.exists() or original != instrument(backup.read_text()):
            raise ValueError("Existing hook differs from its backup")
        print("Already installed", target)
        return
    updated = instrument(original)
    if backup.exists() or helper.exists():
        raise ValueError("Existing backup/helper; inspect before installing")
    helper_source = Path(__file__).with_name("dspark_parity_debug.py")
    backup.write_text(original)
    try:
        shutil.copyfile(helper_source, helper)
        target.write_text(updated)
    except BaseException:
        target.write_text(original)
        backup.unlink(missing_ok=True)
        helper.unlink(missing_ok=True)
        raise
    print("Installed", target, "(disabled unless DSPARK_PARITY_DIR is set)")


if __name__ == "__main__":
    main()
