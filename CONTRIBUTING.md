# Contributing Guidelines

Thank you for your interest in contributing to CLIC!

## Code Style
- Use Black for formatting: `make format`
- Follow PEP 8: `make lint`
- Add type hints where possible

## Testing
- Write unit tests for new features
- Run tests before submitting: `make test`
- Aim for >80% code coverage

## Pull Request Process
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Add tests for new functionality
4. Ensure all tests pass (`make test`)
5. Format code (`make format`)
6. Submit PR with clear description of changes

## Reporting Issues
- Use GitHub Issues for bug reports
- Include reproducible example
- Specify Python/PyTorch versions

## Development Setup
```bash
git clone https://github.com/username/clic.git
cd clic
pip install -e ".[dev]"
make test
```
