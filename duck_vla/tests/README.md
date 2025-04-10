# Duck VLA Test Suite

This directory contains tests for the Duck VLA system.

## Running Tests

### Running All Tests

To run all tests in this directory:

```bash
# From the project root:
python -m pytest duck_vla/tests/

# With more detailed output:
python -m pytest -v duck_vla/tests/
```

### Running Specific Tests

To run a specific test file:

```bash
# Run simulation tests:
python -m pytest duck_vla/tests/test_simulation.py
```

### Running Tests with Simulation Mode

To test the simulation functionality specifically:

```bash
# Run simulation test directly:
python duck_vla/tests/test_simulation.py

# Or using pytest:
python -m pytest duck_vla/tests/test_simulation.py -v
```

## Test Coverage

The test suite covers the following areas:

- Simulation mode functionality
- Motion controller in simulation
- Decision loop with simulation
- (More areas will be added as the project grows)

## Adding New Tests

When adding new tests:

1. Create a new file named `test_*.py` 
2. Import the necessary modules
3. Create a class inheriting from `unittest.TestCase`
4. Add test methods prefixed with `test_`
5. Run the tests to ensure they pass

## Debugging Tests

For debugging tests with more verbose output:

```bash
# Increase logging level
python -m pytest duck_vla/tests/test_simulation.py -v --log-cli-level=DEBUG
``` 