# Contributing

Use Python 3.11–3.13 and uv. Install the locked environment with `uv sync --locked --extra dev`, then run
`uv run ruff check .`, `uv run pytest`, and `uv run donka-scorecard build examples/config.json`.

Keep training data local to the scorecard build. Do not add customer data, credentials, model
outputs containing personal data, or training rows to a pull request. Pack scenarios and sample
data in this repository must be synthetic. Changes to pack output must retain format-v1
compatibility and include an import and scenario run against Studio.
