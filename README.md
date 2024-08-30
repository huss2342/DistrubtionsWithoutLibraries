# Statistical Calculator Script for Casio fx-cg50

This Python script provides custom implementations of various statistical distributions and functions, designed to be used directly on the Casio fx-cg50 calculator's Python environment.

## Purpose

The goal of this script is to provide a set of statistical tools that can be used on the Casio fx-cg50 calculator without relying on external libraries. It implements several probability distributions and normal distribution functions from scratch.

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

1. Transfer the `statistical_calculator.py` file to your Casio fx-cg50 calculator.
2. Run the script in the calculator's Python environment.
3. You will see a menu with the following options:
   ```
   1: distributions
   2: norms
   3: exp(), var()
   ```
4. Choose an option by entering the corresponding number.
5. Follow the prompts to input the required parameters for each calculation.
6. The result will be displayed on the screen.

### Example: Calculating Binomial Distribution

1. Choose option 1 for distributions
2. Select option 3 for dbinom
3. Enter the required parameters (x, size, prob) when prompted
4. The result will be displayed

## Note

This script is designed to be run as a standalone program on the calculator. It provides an interactive interface for performing various statistical calculations without the need for external libraries or additional setup.

## Contributing

If you'd like to contribute to this project, feel free to suggest improvements or additional statistical functions that could be useful for calculator-based computations.