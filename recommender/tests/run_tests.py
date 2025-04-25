#!/usr/bin/env python3
"""
Run all tests for the EU Legal Recommender System with optional coverage reporting.

Usage:
    python run_tests.py           # Run tests without coverage
    python run_tests.py --cov     # Run tests with coverage reporting
    python run_tests.py --html    # Run tests with coverage and generate HTML report
"""
import unittest
import sys
import argparse
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run tests for EU Legal Recommender System')
    parser.add_argument('--cov', action='store_true', help='Run with coverage reporting')
    parser.add_argument('--html', action='store_true', help='Generate HTML coverage report')
    parser.add_argument('--xml', action='store_true', help='Generate XML coverage report')
    parser.add_argument('--source', default='src', help='Source directory to measure coverage for')
    return parser.parse_args()

def run_with_coverage(args):
    """Run tests with coverage reporting."""
    try:
        import coverage
    except ImportError:
        print("Error: coverage package not installed. Run 'pip install coverage' first.")
        sys.exit(1)
    
    # Start coverage measurement
    cov = coverage.Coverage(source=[f"{project_root}/{args.source}"])
    cov.start()
    
    # Run the tests
    run_tests()
    
    # Stop coverage measurement
    cov.stop()
    cov.save()
    
    # Report coverage
    print("\nCoverage Report:")
    cov.report()
    
    # Generate HTML report if requested
    if args.html:
        cov.html_report(directory=f"{project_root}/htmlcov")
        print(f"\nHTML coverage report generated in {project_root}/htmlcov/index.html")
    
    # Generate XML report if requested
    if args.xml:
        cov.xml_report(outfile=f"{project_root}/coverage.xml")
        print(f"\nXML coverage report generated in {project_root}/coverage.xml")

def run_tests():
    """Run all tests without coverage reporting."""
    # Discover and run all tests
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover(start_dir=Path(__file__).parent, pattern='test_*.py')
    
    # Run the tests
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    return result.wasSuccessful()

if __name__ == '__main__':
    args = parse_args()
    
    if args.cov or args.html or args.xml:
        run_with_coverage(args)
    else:
        success = run_tests()
        # Exit with non-zero code if there were failures
        sys.exit(not success)
