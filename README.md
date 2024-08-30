# Statistical Calculator for Casio fx-cg50

This project provides a Python script with custom implementations of various statistical distributions and functions, designed to be used on the Casio fx-cg50 calculator's Python environment without the need for external libraries.

## Purpose

The goal of this project is to provide a set of statistical tools that can be used on the Casio fx-cg50 calculator, which has limited library support. By implementing these functions from scratch, users can perform advanced statistical calculations directly on their calculator.

## Features

- Probability distributions:
  - Negative Binomial (dnbinom, pnbinom)
  - Binomial (dbinom, pbinom)
  - Hypergeometric (dhyper, phyper)
  - Poisson (ppois)
- Normal distribution functions:
  - Cumulative Distribution Function (pnorm)
  - Quantile Function (qnorm)
- Helper functions for calculations

## Usage

1. Transfer the `main.py` file to your Casio fx-cg50 calculator.
2. Run the script in the calculator's Python environment.
3. Follow the on-screen prompts to select the type of calculation you want to perform.
4. Input the required parameters as requested.
5. The result will be displayed on the screen.

## Development and Testing

This project includes a set of unit tests to verify the accuracy of the implemented functions. To run the tests:

1. Ensure you have Python installed on your computer.
2. Navigate to the project directory in a terminal.
3. Run the command: `python -m unittest unit_tests.test_statistical_functions`

Note: The tests are designed with some leniency to account for slight variations in floating-point arithmetic that may occur on different systems.

## License and Usage

This code is open and free for anyone to use, modify, and distribute. There are no restrictions on its usage. Feel free to incorporate it into your projects or make improvements as you see fit.

## Contributing

Contributions to improve the implementations or add new statistical functions are welcome. If you make improvements, consider submitting a pull request to benefit other users.

## Disclaimer

While efforts have been made to ensure accuracy, these implementations may not match the precision of specialized statistical software. They are intended for educational purposes and general use on a graphing calculator.