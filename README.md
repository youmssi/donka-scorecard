# donka-scorecard

Build a binary statistical scorecard with [OptBinning](https://github.com/guillermo-navas-palencia/optbinning)
and export it as a Donka pack (format v1). The tool runs on the data scientist's machine;
training data is not uploaded to Studio or copied into the generated pack.

## Quick start

```sh
uv sync --extra dev
uv run donka-scorecard build examples/config.json
```

The example writes `examples/dist/statistical-scorecard-example/` and
`examples/model-report.json`. Import that directory or zip its contents and use Studio's
**Projects → From a pack → Pack file** flow. Studio evaluates the generated sample scenarios
against the imported graph. The example data is deterministic synthetic data and the example
policy must not be used for lending.

## Configuration

The version-1 JSON config names a CSV file, binary target and labels, development and out-of-time
date windows, and the feature list. Each feature has a unique lowercase name, a source column,
an input field, and `numeric` or `categorical` type. Categorical features also list the allowed
input values. Binning sizes and points scaling are explicit. Keep input field names and allowed
values aligned with the source application contract.

The development window fits bins and logistic regression. The later out-of-time window measures
Gini and KS and supplies real rows to the pass scenarios. The report includes bin counts and
points, target/window metadata, sample sizes, and accepted-only/small-sample limitations. A run
stops before fitting when development has fewer bad outcomes than `minimum_bad_outcomes`.

## Boundaries

This tool does not perform reject inference, upload data, or promote releases. A lender must
validate target definitions, representativeness, fairness, calibration, performance, local
regulations, and policy thresholds before use. Use synthetic or approved local data and keep
training files out of version control.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md). Licensed under Apache-2.0.
