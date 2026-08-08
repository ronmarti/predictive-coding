# Python Code Style Guide

This project uses Python with the following styles:

- Use snake_case for variable and function names.
- Use PascalCase for class names.
- Use 4 spaces for indentation.
- Limit lines to 79 characters.
- Use docstrings to describe the purpose of classes and functions.
- Use type hints for function parameters and return types.
- Use f-strings for string formatting, e.g., `f"Hello, {name}!"`.
- Avoid using global variables; prefer passing parameters and returning values.
- Use list comprehensions and generator expressions where appropriate.
- Handle exceptions gracefully and log errors using the logging module.
- Prefer using pathlib for file system paths over os.path.
- Follow PEP 8 guidelines for code formatting and organization.
- Prefer dependency injection for better testability and modularity.
- All configuration values should be read from a configuration file or environment variables, not hardcoded in the code.
- Single responsibility principle: each class and function should have a single responsibility or purpose.
- One class per file, and the file name should match the class name in snake_case.
- Avoid using magic numbers; define constants with meaningful names instead.
- Large vectorized operations should be performed using libraries like NumPy for better performance.
- Comment complex logic and non-obvious code sections to improve readability and maintainability.
- Prefer using asyncio tasks instead of threads for concurrent execution in I/O-bound scenarios.
- Use async/await for asynchronous operations when dealing with I/O-bound tasks.
- Use the latest approach for asyncio (e.g., asyncio.run() for the main entry point) and avoid deprecated patterns.
- Avoid using low level python attributes like __dict__ or __class__ unless necessary for advanced use cases.
- Avoid runtime imports of libraries that are already declared as dependencies in pyproject.toml, as this can lead to import errors and makes it harder to track dependencies. Instead, import all necessary libraries at the top of the file.
- Have environment variables all managed close together in one class at the beginning of the application.
- Avoid nesting functions inside other functions unless necessary for closures or decorators. Instead, define functions at the module level for better readability and testability.
- Avoid using `functools.partial` as it lowers readability.
- Use flat layout, as documented online here: https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/

## Directory Structure
- `application/`: Contains the entry point to the application.
  - `models/`: Contains the data models, filters, distributors, and reducers used in the processing.
  - `logic/`: Contains the business logic of higher layer, such as the orchestrator that manages the execution of the processing.
  - `utils/`: Contains utility functions and classes, such as logging configuration or resource monitoring. Use only if no other directory is suitable and consider creating a new directory if the utilities are substantial and can be categorized further.

## Context consumption
Always use specific workspace search tools rather than reading entire files into context unless requested. 
If a task is too large for the current context window, break it into smaller sub-tasks.

# Documentation Resources

## Opinionated Guideline for Python
- Duarte: https://duarteocarmo.com/blog/opinionated-python-boilerplate

## Python project layout
- https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/

Always cite documentation when explaining concepts.