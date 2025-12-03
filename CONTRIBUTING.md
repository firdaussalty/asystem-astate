# How to contribute to Astate

## Finding good first issues

See [Good First Issues](https://github.com/inclusionAI/astate/contribute).

## How to create an issue

Create an issue with [this form](https://github.com/inclusionAI/astate/issues/new/choose).

## How to title your PR

Generally we follows the [Conventional Commits](https://www.conventionalcommits.org/) for pull request titles,
since we will squash and merge the PR and use the PR title as the first line of commit message.

For example, here are good PR titles:

- feat: support xxx feature
- fix: blablabla
- chore: remove useless yyy file
- docs: add api doc for xxx method

For more details, please check [pr-lint.yml](./.github/workflows/pr-lint.yml).

## Testing

For environmental requirements, please check [DEVELOPMENT.md](DEVELOPMENT.md).

Run tests:

```bash
make test
```

## Code Style

Run all checks: `bash format.sh`.

## Development

For more information, please refer to [Development Guide](DEVELOPMENT.md).