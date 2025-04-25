# Development Documentation

This directory contains documentation for developers working on the EU Legal Recommender system.

## Development Setup

1. Clone the repository:
   ```
   git clone https://github.com/your-org/eu-legal-recommender.git
   cd eu-legal-recommender
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install development dependencies:
   ```
   pip install -e ".[dev]"
   ```

4. Set up pre-commit hooks:
   ```
   pre-commit install
   ```

## Code Structure

The codebase follows the structure outlined in `PROJECT_STRUCTURE.md` at the root of the repository.

## Testing

Run the test suite:
```
python -m tests.run_tests
```

Run tests with coverage:
```
python -m tests.run_tests --cov --html
```

## Code Style

The project follows the style guide defined in `STYLE_GUIDE.md`. Key points:

- Use Black for code formatting
- Follow PEP 8 naming conventions
- Use type hints for all functions and methods
- Write comprehensive docstrings in Google style

## Pull Request Process

1. Create a new branch for your feature or bugfix
2. Implement your changes with tests
3. Ensure all tests pass and coverage is maintained
4. Submit a pull request with a clear description of changes
5. Address any review comments

## Release Process

Information about the release process will be added here.
