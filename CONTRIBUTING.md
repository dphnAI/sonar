## Contributing Guidelines for Sonar Engine

Sonar Engine welcomes any and all contributors to submit fixes, features, issues - no matter the scope.

### Code of Ethics

If you wish, you can read the [Sonar Engine Code of Conduct](./CODE_OF_CONDUCT.md). You don't need to follow it! :)

### Setup

For development, you will need to build the engine from source.

```sh
pip install -e .
pip install -r requirements/dev.txt
```

Make sure you build in editable mode with `-e`.

Our test units at the moment are incomplete, so you can skip test units for now.

Install the pre-commit hooks once so the linters and formatters run on every commit:

```sh
pre-commit install
```

To run them by hand on every file:

```sh
pre-commit run --all-files
```

Thank you for your interest in Sonar Engine!
