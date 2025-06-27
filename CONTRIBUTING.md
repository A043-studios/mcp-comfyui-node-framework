# Contributing to MCP ComfyUI Node Framework

Thank you for your interest in contributing to the MCP ComfyUI Node Framework! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

### Reporting Issues

1. **Search existing issues** to avoid duplicates
2. **Use issue templates** when available
3. **Provide detailed information**:
   - Framework version
   - Operating system
   - Python/Node.js versions
   - Steps to reproduce
   - Expected vs actual behavior
   - Error messages and logs

### Suggesting Features

1. **Check the roadmap** in [CHANGELOG.md](CHANGELOG.md)
2. **Open a discussion** before creating an issue
3. **Describe the use case** and benefits
4. **Consider implementation complexity**

### Contributing Code

1. **Fork the repository**
2. **Create a feature branch** from `main`
3. **Make your changes** following our guidelines
4. **Add tests** for new functionality
5. **Update documentation** as needed
6. **Submit a pull request**

## 🛠️ Development Setup

### Prerequisites

- Python 3.8+
- Node.js 18+
- Git

### Setup Steps

```bash
# Clone your fork
git clone https://github.com/your-username/mcp-comfyui-node-framework.git
cd mcp-comfyui-node-framework

# Set up Python environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
npm install

# Set up configuration
cp config/mcp-config.template.json config/mcp-config.json
# Edit config/mcp-config.json with your API keys

# Run tests
python -m pytest tests/
npm test
```

### Development Workflow

1. **Create a branch** for your feature/fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make changes** following our coding standards

3. **Test your changes**:
   ```bash
   # Run Python tests
   python -m pytest tests/ -v
   
   # Run integration tests
   python tests/test_integration.py
   
   # Test specific functionality
   python tests/test_integration.py --class TestNodeValidation
   ```

4. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat: add new feature description"
   ```

5. **Push and create PR**:
   ```bash
   git push origin feature/your-feature-name
   ```

## 📝 Coding Standards

### Python Code Style

- **Follow PEP 8** with line length of 88 characters
- **Use type hints** for function parameters and return values
- **Write docstrings** for all public functions and classes
- **Use meaningful variable names**
- **Keep functions focused** and under 50 lines when possible

Example:
```python
def generate_node_code(
    analysis: Dict[str, Any], 
    quality_level: str = "production"
) -> Dict[str, Any]:
    """
    Generate ComfyUI node code from analysis results.
    
    Args:
        analysis: Analysis results from content analysis
        quality_level: Target quality level (draft/development/production)
        
    Returns:
        Dictionary containing generated code and metadata
        
    Raises:
        NodeGenerationError: If code generation fails
    """
    # Implementation here
    pass
```

### TypeScript/JavaScript Code Style

- **Use TypeScript** for new code
- **Follow Prettier** formatting
- **Use meaningful names** and avoid abbreviations
- **Add JSDoc comments** for public functions
- **Prefer async/await** over promises

### Documentation Style

- **Use Markdown** for all documentation
- **Include code examples** for API functions
- **Keep explanations clear** and concise
- **Update relevant docs** when changing functionality
- **Use proper headings** and structure

## 🧪 Testing Guidelines

### Test Types

1. **Unit Tests**: Test individual functions and classes
2. **Integration Tests**: Test component interactions
3. **End-to-End Tests**: Test complete workflows
4. **Node Tests**: Validate generated ComfyUI nodes

### Writing Tests

```python
import unittest
from src.your_module import YourClass

class TestYourClass(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.instance = YourClass()
    
    def test_functionality(self):
        """Test specific functionality"""
        result = self.instance.method()
        self.assertEqual(result, expected_value)
    
    def test_error_handling(self):
        """Test error conditions"""
        with self.assertRaises(ExpectedError):
            self.instance.method_that_should_fail()
```

### Test Coverage

- **Aim for 80%+ coverage** for new code
- **Test both success and failure cases**
- **Include edge cases** and boundary conditions
- **Mock external dependencies** (LLM APIs, web requests)

## 📚 Documentation Guidelines

### Code Documentation

- **Document all public APIs** with docstrings
- **Include usage examples** in docstrings
- **Explain complex algorithms** with comments
- **Keep comments up-to-date** with code changes

### User Documentation

- **Update README.md** for new features
- **Add examples** to the examples directory
- **Update API.md** for API changes
- **Include configuration** instructions

### Generated Node Documentation

- **Include installation instructions**
- **Provide usage examples**
- **Document all parameters**
- **Add troubleshooting section**

## 🔄 Pull Request Process

### Before Submitting

1. **Ensure tests pass** locally
2. **Update documentation** as needed
3. **Check code style** with linters
4. **Rebase on latest main** if needed

### PR Description

Include:
- **Clear description** of changes
- **Motivation** for the changes
- **Testing performed**
- **Breaking changes** (if any)
- **Related issues** (if any)

### Review Process

1. **Automated checks** must pass
2. **Code review** by maintainers
3. **Testing** in different environments
4. **Documentation review**
5. **Final approval** and merge

## 🏷️ Commit Message Format

Use conventional commits format:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test additions/changes
- `chore`: Maintenance tasks

Examples:
```
feat(agents): add new research agent for arXiv papers
fix(llm): handle API timeout errors gracefully
docs(api): update configuration examples
```

## 🎯 Areas for Contribution

### High Priority

- **New node templates** for different use cases
- **Additional source types** (documentation, APIs)
- **Performance optimizations** for large repositories
- **Better error handling** and user feedback
- **Test coverage improvements**

### Medium Priority

- **UI/UX improvements** for configuration
- **Additional LLM providers** integration
- **Workflow optimization** features
- **Documentation enhancements**
- **Example nodes** for popular repositories

### Low Priority

- **Code style improvements**
- **Minor bug fixes**
- **Documentation typos**
- **Dependency updates**

## 🆘 Getting Help

### Communication Channels

- **GitHub Discussions**: General questions and ideas
- **GitHub Issues**: Bug reports and feature requests
- **Code Reviews**: Technical discussions in PRs

### Resources

- [Framework Documentation](README.md)
- [API Reference](docs/API.md)
- [Configuration Guide](docs/CONFIGURATION.md)
- [Example Nodes](examples/README.md)

## 📄 License

By contributing to this project, you agree that your contributions will be licensed under the MIT License.

## 🙏 Recognition

Contributors will be:
- **Listed in CONTRIBUTORS.md**
- **Mentioned in release notes**
- **Credited in documentation**

Thank you for helping make the MCP ComfyUI Node Framework better! 🚀
