#!/bin/bash

echo "================================================"
echo "TabTransformer Test Suite"
echo "================================================"

# Run unit tests
echo -e "\n>>> Running Unit Tests..."
python src/test_tab_transformer.py
UNIT_TEST=$?

# Run integration test
echo -e "\n>>> Running Integration Test..."
python test_integration.py
INTEGRATION_TEST=$?

# Summary
echo -e "\n================================================"
echo "Test Summary:"
echo "================================================"
[ $UNIT_TEST -eq 0 ] && echo "✓ Unit Tests: PASSED" || echo "✗ Unit Tests: FAILED"
[ $INTEGRATION_TEST -eq 0 ] && echo "✓ Integration Test: PASSED" || echo "✗ Integration Test: FAILED"

if [ $UNIT_TEST -eq 0 ] && [ $INTEGRATION_TEST -eq 0 ]; then
    echo -e "\n✓ All tests passed!"
    exit 0
else
    echo -e "\n✗ Some tests failed!"
    exit 1
fi
