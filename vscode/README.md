# ASM-Lang VS Code extension (local)

This extension provides minimal language support and run commands for ASM-Lang (ASMLN) bundled with this repository.

Features:
- Registers the `asmln` language and associates `.asmln` files.
- Basic syntax highlighting for comments, strings, binary integer literals, keywords and builtins.
- Two commands:
  - `ASM-Lang: Run File` — runs the current file using the workspace `asmln.py` in an integrated terminal.
  - `ASM-Lang: Start REPL` — starts the REPL by launching the interpreter with no file.

Configuration:
- `asmln.pythonPath` — path to the Python interpreter executable (default: `python`).

Usage:
1. Open the workspace root (this repository) in VS Code.
2. Open an `.asmln` file.
3. Run the command palette (Ctrl+Shift+P) and pick `ASM-Lang: Run File` or `ASM-Lang: Start REPL`.

Notes:
- The extension expects `asmln.py` to live in the workspace root. If your workspace layout differs, update the command or set a full path in `asmln.pythonPath`.
