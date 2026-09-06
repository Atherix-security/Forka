# Contributing to Forka

Thanks for helping build Forka. The project is intentionally early and welcomes design discussion, bug reports, documentation, tests, and focused code contributions.

## Development

```bash
git clone https://github.com/Atherix-security/Forka.git
cd Forka
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest
ruff check .
```

Keep the core lightweight. New mandatory dependencies need a clear reason, especially if they make CPU-only installation harder.

For substantial features, open an issue before a large pull request so the design can be discussed first.
