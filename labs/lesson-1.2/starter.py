"""
Lab 1.2: Statistics & Probability for ML
==========================================
Complete the exercises below. Each function has instructions in its docstring.
Run this file to execute all exercises: python starter.py
"""

import numpy as np
import matplotlib.pyplot as plt


# ============================================================================
# Exercise 1: Distribution Explorer
# ============================================================================

def exercise_1():
    """
    Generate and visualize different probability distributions.

    Tasks:
    1. Generate 10,000 samples from:
       a) Normal distributions with (mean=0, std=1), (mean=0, std=3), (mean=5, std=1)
       b) Uniform distributions with (low=0, high=1), (low=-5, high=5)
       c) Binomial distributions with (n=10, p=0.5), (n=10, p=0.2), (n=10, p=0.8)
    2. Plot histograms in a 3x3 grid
    3. Add vertical lines for mean and median on each subplot
    """
    np.random.seed(42)

    # TODO: Generate samples
    # normal_1 = np.random.normal(0, 1, 10000)
    # normal_2 = np.random.normal(0, 3, 10000)
    # normal_3 = np.random.normal(5, 1, 10000)
    # uniform_1 = np.random.uniform(0, 1, 10000)
    # uniform_2 = np.random.uniform(-5, 5, 10000)
    # binom_1 = np.random.binomial(10, 0.5, 10000)
    # binom_2 = np.random.binomial(10, 0.2, 10000)
    # binom_3 = np.random.binomial(10, 0.8, 10000)

    # TODO: Create a 3x3 grid of histograms
    # distributions = [
    #     (normal_1, "Normal(0, 1)"), (normal_2, "Normal(0, 3)"), (normal_3, "Normal(5, 1)"),
    #     (uniform_1, "Uniform(0, 1)"), (uniform_2, "Uniform(-5, 5)"), (binom_1, "Binom(10, 0.5)"),
    #     (binom_2, "Binom(10, 0.2)"), (binom_3, "Binom(10, 0.8)"), (normal_1, "placeholder")
    # ]
    #
    # fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    # for ax, (data, title) in zip(axes.flat, distributions):
    #     ax.hist(data, bins=40, density=True, alpha=0.7, color='steelblue')
    #     ax.axvline(np.mean(data), color='red', linestyle='--', label=f'Mean={np.mean(data):.2f}')
    #     ax.axvline(np.median(data), color='green', linestyle='--', label=f'Median={np.median(data):.2f}')
    #     ax.set_title(title)
    #     ax.legend(fontsize=8)
    # plt.tight_layout()
    # plt.show()

    print("Exercise 1: Complete the TODOs above\n")


# ============================================================================
# Exercise 2: Statistics Calculator
# ============================================================================

def exercise_2():
    """
    Compute descriptive statistics and detect outliers using z-scores.

    Tasks:
    1. Generate a dataset of 500 "salaries" from a normal distribution
       (mean=50000, std=15000), then add 5 outliers (e.g., 200000+)
    2. Compute: mean, median, variance, std dev, min, max
    3. Compute z-scores for every data point
    4. Identify outliers where |z| > 3
    5. Plot histogram with outliers highlighted
    6. Compare mean/median before and after removing outliers
    """
    np.random.seed(42)

    # TODO: Generate salary data with outliers
    # salaries = np.random.normal(50000, 15000, 500)
    # outliers = np.array([200000, 250000, 300000, 280000, 220000])
    # salaries = np.concatenate([salaries, outliers])

    # TODO: Compute descriptive statistics
    # print(f"Mean:     ${np.mean(salaries):,.0f}")
    # print(f"Median:   ${np.median(salaries):,.0f}")
    # print(f"Std Dev:  ${np.std(salaries):,.0f}")
    # print(f"Variance: {np.var(salaries):,.0f}")
    # print(f"Min:      ${np.min(salaries):,.0f}")
    # print(f"Max:      ${np.max(salaries):,.0f}")

    # TODO: Compute z-scores and find outliers
    # z_scores = (salaries - np.mean(salaries)) / np.std(salaries)
    # outlier_mask = np.abs(z_scores) > 3
    # print(f"\nOutliers detected (|z| > 3): {np.sum(outlier_mask)}")
    # print(f"Outlier values: {salaries[outlier_mask]}")

    # TODO: Plot histogram highlighting outliers
    # plt.figure(figsize=(10, 5))
    # plt.hist(salaries[~outlier_mask], bins=40, alpha=0.7, color='steelblue', label='Normal')
    # plt.hist(salaries[outlier_mask], bins=10, alpha=0.7, color='red', label='Outliers')
    # plt.axvline(np.mean(salaries), color='red', linestyle='--', label=f'Mean=${np.mean(salaries):,.0f}')
    # plt.axvline(np.median(salaries), color='green', linestyle='--', label=f'Median=${np.median(salaries):,.0f}')
    # plt.legend()
    # plt.title('Salary Distribution with Outliers')
    # plt.xlabel('Salary ($)')
    # plt.ylabel('Count')
    # plt.show()

    # TODO: Compare stats with and without outliers
    # clean = salaries[~outlier_mask]
    # print(f"\nAfter removing outliers:")
    # print(f"Mean:   ${np.mean(clean):,.0f} (was ${np.mean(salaries):,.0f})")
    # print(f"Median: ${np.median(clean):,.0f} (was ${np.median(salaries):,.0f})")

    print("Exercise 2: Complete the TODOs above\n")


# ============================================================================
# Exercise 3: Bayes' Theorem in Practice
# ============================================================================

def exercise_3():
    """
    Explore how base rate affects posterior probability.

    Tasks:
    1. Implement Bayes' theorem as a function
    2. Fix sensitivity=0.95 and specificity=0.90
    3. Vary prevalence from 0.01 to 0.50 (50 values)
    4. Compute P(Disease | Positive) for each prevalence
    5. Plot prevalence vs posterior probability
    6. Add a horizontal line at 50% and annotate where it crosses
    """

    # TODO: Implement Bayes' theorem
    def bayes_positive_test(prevalence, sensitivity, specificity):
        """
        Compute P(Disease | Positive Test) using Bayes' theorem.

        Args:
            prevalence: P(Disease) — base rate
            sensitivity: P(Positive | Disease) — true positive rate
            specificity: P(Negative | No Disease) — true negative rate

        Returns:
            P(Disease | Positive Test)
        """
        # p_positive = sensitivity * prevalence + (1 - specificity) * (1 - prevalence)
        # return (sensitivity * prevalence) / p_positive
        pass

    # TODO: Vary prevalence and compute posteriors
    # sensitivity = 0.95
    # specificity = 0.90
    # prevalences = np.linspace(0.01, 0.50, 50)
    # posteriors = [bayes_positive_test(p, sensitivity, specificity) for p in prevalences]

    # TODO: Plot
    # plt.figure(figsize=(10, 6))
    # plt.plot(prevalences * 100, np.array(posteriors) * 100, 'b-', linewidth=2)
    # plt.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% threshold')
    # plt.xlabel('Disease Prevalence (%)')
    # plt.ylabel('P(Disease | Positive Test) (%)')
    # plt.title('How Base Rate Affects Diagnostic Certainty\n(Sensitivity=95%, Specificity=90%)')
    # plt.grid(True, alpha=0.3)
    # plt.legend()
    # plt.show()

    # TODO: Print the crossover point
    # for prev, post in zip(prevalences, posteriors):
    #     if post >= 0.5:
    #         print(f"Posterior reaches 50% at prevalence ≈ {prev:.1%}")
    #         break

    print("Exercise 3: Complete the TODOs above\n")


# ============================================================================
# Exercise 4: Correlation Investigation
# ============================================================================

def exercise_4():
    """
    Generate data with known correlations and visualize.

    Tasks:
    1. Generate 3 pairs of variables:
       a) Strong positive correlation (r ≈ 0.9)
       b) Strong negative correlation (r ≈ -0.8)
       c) No correlation (r ≈ 0)
    2. Compute the correlation matrix
    3. Create scatter plots for each pair
    4. Print the full correlation matrix
    """
    np.random.seed(42)

    # TODO: Generate correlated data
    # Hint: To generate correlated variables, start with one and derive the other
    # n = 300
    # x = np.random.normal(0, 1, n)
    #
    # # Strong positive correlation
    # y_pos = 0.9 * x + 0.1 * np.random.normal(0, 1, n)  # r ≈ 0.9
    #
    # # Strong negative correlation
    # y_neg = -0.8 * x + 0.2 * np.random.normal(0, 1, n)  # r ≈ -0.8
    #
    # # No correlation
    # y_none = np.random.normal(0, 1, n)  # r ≈ 0

    # TODO: Compute correlations
    # print(f"Positive correlation: r = {np.corrcoef(x, y_pos)[0,1]:.3f}")
    # print(f"Negative correlation: r = {np.corrcoef(x, y_neg)[0,1]:.3f}")
    # print(f"No correlation:       r = {np.corrcoef(x, y_none)[0,1]:.3f}")

    # TODO: Create scatter plots
    # fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    #
    # for ax, y_data, title, color in zip(axes,
    #     [y_pos, y_neg, y_none],
    #     ['Strong Positive', 'Strong Negative', 'No Correlation'],
    #     ['steelblue', 'coral', 'gray']):
    #     ax.scatter(x, y_data, alpha=0.4, s=20, color=color)
    #     r = np.corrcoef(x, y_data)[0, 1]
    #     ax.set_title(f'{title}\nr = {r:.3f}')
    #     ax.set_xlabel('x')
    #     ax.set_ylabel('y')
    #     ax.grid(True, alpha=0.3)
    #
    # plt.tight_layout()
    # plt.show()

    # TODO: Full correlation matrix
    # all_vars = np.column_stack([x, y_pos, y_neg, y_none])
    # corr_matrix = np.corrcoef(all_vars.T)
    # print("\nFull correlation matrix:")
    # labels = ['x', 'y_pos', 'y_neg', 'y_none']
    # print(f"{'':>8}", end='')
    # for label in labels:
    #     print(f"{label:>10}", end='')
    # print()
    # for i, label in enumerate(labels):
    #     print(f"{label:>8}", end='')
    #     for j in range(len(labels)):
    #         print(f"{corr_matrix[i, j]:>10.3f}", end='')
    #     print()

    print("Exercise 4: Complete the TODOs above\n")


# ============================================================================
# Run all exercises
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Lab 1.2: Statistics & Probability for ML")
    print("=" * 60)
    print()

    print("--- Exercise 1: Distribution Explorer ---")
    exercise_1()

    print("--- Exercise 2: Statistics Calculator ---")
    exercise_2()

    print("--- Exercise 3: Bayes' Theorem ---")
    exercise_3()

    print("--- Exercise 4: Correlation Investigation ---")
    exercise_4()

    print("Done! Uncomment the TODO sections to complete each exercise.")
