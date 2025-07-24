import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from collections import Counter

class ProbabilityToolkit:
    """
    Toolkit for understanding probability concepts in machine learning
    """
    
    @staticmethod
    def bayes_theorem_demo():
        """
        Demonstrate Bayes' theorem with a medical diagnosis example
        """
        print("Bayes' Theorem: Medical Diagnosis Example")
        print("=" * 45)
        
        # Problem: Disease testing
        # - Disease prevalence: 1% of population
        # - Test accuracy: 99% (correctly identifies sick and healthy)
        
        prior_disease = 0.01  # P(Disease) = 1%
        prior_healthy = 0.99  # P(Healthy) = 99%
        
        likelihood_pos_given_disease = 0.99  # P(Positive|Disease) = 99%
        likelihood_pos_given_healthy = 0.01  # P(Positive|Healthy) = 1%
        
        # Calculate marginal probability P(Positive)
        marginal_positive = (likelihood_pos_given_disease * prior_disease + 
                           likelihood_pos_given_healthy * prior_healthy)
        
        # Apply Bayes' theorem: P(Disease|Positive)
        posterior_disease = (likelihood_pos_given_disease * prior_disease) / marginal_positive
        
        print(f"Prior probability of disease: {prior_disease:.1%}")
        print(f"Test accuracy: {likelihood_pos_given_disease:.1%}")
        print(f"Probability of positive test: {marginal_positive:.1%}")
        print(f"Probability of disease given positive test: {posterior_disease:.1%}")
        print(f"\nSurprising result: Even with 99% accurate test,")
        print(f"positive result only means {posterior_disease:.1%} chance of disease!")
        
        return posterior_disease
    
    @staticmethod
    def distribution_gallery():
        """
        Visualize common probability distributions used in ML
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Common Probability Distributions in Machine Learning', fontsize=16)
        
        # Normal Distribution
        x_norm = np.linspace(-4, 4, 1000)
        axes[0,0].plot(x_norm, stats.norm.pdf(x_norm, 0, 1), 'b-', label='μ=0, σ=1')
        axes[0,0].plot(x_norm, stats.norm.pdf(x_norm, 1, 0.5), 'r-', label='μ=1, σ=0.5')
        axes[0,0].set_title('Normal Distribution')
        axes[0,0].set_xlabel('x')
        axes[0,0].set_ylabel('Probability Density')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Binomial Distribution
        n, p = 20, 0.3
        x_binom = np.arange(0, n+1)
        axes[0,1].bar(x_binom, stats.binom.pmf(x_binom, n, p), alpha=0.7)
        axes[0,1].set_title(f'Binomial Distribution (n={n}, p={p})')
        axes[0,1].set_xlabel('Number of Successes')
        axes[0,1].set_ylabel('Probability')
        axes[0,1].grid(True, alpha=0.3)
        
        # Poisson Distribution
        mu_vals = [1, 4, 10]
        x_poisson = np.arange(0, 20)
        colors = ['blue', 'red', 'green']
        for mu, color in zip(mu_vals, colors):
            axes[0,2].bar(x_poisson, stats.poisson.pmf(x_poisson, mu), 
                         alpha=0.6, label=f'λ={mu}', color=color)
        axes[0,2].set_title('Poisson Distribution')
        axes[0,2].set_xlabel('Count')
        axes[0,2].set_ylabel('Probability')
        axes[0,2].legend()
        axes[0,2].grid(True, alpha=0.3)
        
        # Exponential Distribution
        x_exp = np.linspace(0, 5, 1000)
        lambdas = [0.5, 1, 2]
        for lam in lambdas:
            axes[1,0].plot(x_exp, stats.expon.pdf(x_exp, scale=1/lam), 
                          label=f'λ={lam}')
        axes[1,0].set_title('Exponential Distribution')
        axes[1,0].set_xlabel('x')
        axes[1,0].set_ylabel('Probability Density')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Beta Distribution
        x_beta = np.linspace(0, 1, 1000)
        alpha_beta_pairs = [(0.5, 0.5), (2, 5), (5, 2)]
        for alpha, beta in alpha_beta_pairs:
            axes[1,1].plot(x_beta, stats.beta.pdf(x_beta, alpha, beta), 
                          label=f'α={alpha}, β={beta}')
        axes[1,1].set_title('Beta Distribution')
        axes[1,1].set_xlabel('x')
        axes[1,1].set_ylabel('Probability Density')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        # Uniform Distribution
        x_uniform = np.linspace(-1, 3, 1000)
        axes[1,2].plot(x_uniform, stats.uniform.pdf(x_uniform, 0, 2), 'b-', linewidth=3)
        axes[1,2].set_title('Uniform Distribution (0, 2)')
        axes[1,2].set_xlabel('x')
        axes[1,2].set_ylabel('Probability Density')
        axes[1,2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

class StatisticalInference:
    """
    Tools for statistical inference and hypothesis testing
    """
    
    @staticmethod
    def central_limit_theorem_demo(n_samples=1000, sample_sizes=[1, 5, 30]):
        """
        Demonstrate Central Limit Theorem with different sample sizes
        """
        # Generate data from non-normal distribution (exponential)
        population = np.random.exponential(scale=2, size=10000)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Central Limit Theorem Demonstration', fontsize=16)
        
        # Plot original population
        axes[0,0].hist(population, bins=50, alpha=0.7, density=True)
        axes[0,0].set_title('Original Population (Exponential)')
        axes[0,0].set_xlabel('Value')
        axes[0,0].set_ylabel('Density')
        
        colors = ['blue', 'red', 'green']
        for i, (n, color) in enumerate(zip(sample_sizes, colors)):
            # Generate sampling distribution
            sample_means = []
            for _ in range(n_samples):
                sample = np.random.choice(population, size=n)
                sample_means.append(np.mean(sample))
            
            # Plot sampling distribution
            row, col = (0, 1) if i == 0 else (1, i-1)
            axes[row, col].hist(sample_means, bins=30, alpha=0.7, density=True, color=color)
            axes[row, col].set_title(f'Sample Means (n={n})')
            axes[row, col].set_xlabel('Sample Mean')
            axes[row, col].set_ylabel('Density')
            
            # Overlay normal approximation
            mean_of_means = np.mean(sample_means)
            std_of_means = np.std(sample_means)
            x_norm = np.linspace(min(sample_means), max(sample_means), 100)
            y_norm = stats.norm.pdf(x_norm, mean_of_means, std_of_means)
            axes[row, col].plot(x_norm, y_norm, 'k--', linewidth=2, label='Normal Approx')
            axes[row, col].legend()
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def hypothesis_test_demo(sample1, sample2, alpha=0.05):
        """
        Perform t-test to compare two samples
        """
        print("Hypothesis Testing: Two-Sample T-Test")
        print("=" * 40)
        
        # Calculate descriptive statistics
        mean1, std1, n1 = np.mean(sample1), np.std(sample1, ddof=1), len(sample1)
        mean2, std2, n2 = np.mean(sample2), np.std(sample2, ddof=1), len(sample2)
        
        print(f"Sample 1: mean={mean1:.3f}, std={std1:.3f}, n={n1}")
        print(f"Sample 2: mean={mean2:.3f}, std={std2:.3f}, n={n2}")
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(sample1, sample2)
        
        print(f"\nHypothesis Test Results:")
        print(f"H₀: μ₁ = μ₂ (no difference in means)")
        print(f"H₁: μ₁ ≠ μ₂ (difference in means)")
        print(f"α = {alpha}")
        print(f"t-statistic: {t_stat:.4f}")
        print(f"p-value: {p_value:.4f}")
        
        # Conclusion
        if p_value < alpha:
            print(f"Result: Reject H₀ (p < α)")
            print(f"Conclusion: Significant difference between groups")
        else:
            print(f"Result: Fail to reject H₀ (p ≥ α)")
            print(f"Conclusion: No significant difference between groups")
        
        # Confidence interval for difference
        diff_mean = mean1 - mean2
        pooled_se = np.sqrt(std1**2/n1 + std2**2/n2)
        df = n1 + n2 - 2
        t_critical = stats.t.ppf(1 - alpha/2, df)
        ci_lower = diff_mean - t_critical * pooled_se
        ci_upper = diff_mean + t_critical * pooled_se
        
        print(f"\n{100*(1-alpha):.0f}% CI for mean difference: [{ci_lower:.3f}, {ci_upper:.3f}]")
        
        return t_stat, p_value

class InformationTheory:
    """
    Information theory concepts for machine learning
    """
    
    @staticmethod
    def entropy_calculation(probabilities):
        """
        Calculate entropy of a probability distribution
        """
        # Remove zero probabilities to avoid log(0)
        p = np.array(probabilities)
        p = p[p > 0]
        
        # Calculate entropy: H(X) = -Σ p(x) log₂ p(x)
        entropy = -np.sum(p * np.log2(p))
        return entropy
    
    @staticmethod
    def kl_divergence(p, q):
        """
        Calculate KL divergence between two distributions
        """
        p = np.array(p)
        q = np.array(q)
        
        # Avoid division by zero
        mask = (p > 0) & (q > 0)
        p = p[mask]
        q = q[mask]
        
        # KL(P||Q) = Σ p(x) log(p(x)/q(x))
        kl_div = np.sum(p * np.log(p / q))
        return kl_div
    
    @staticmethod
    def information_theory_demo():
        """
        Demonstrate entropy and KL divergence concepts
        """
        print("Information Theory Demonstration")
        print("=" * 35)
        
        # Different probability distributions
        uniform = [0.25, 0.25, 0.25, 0.25]  # Maximum entropy
        skewed = [0.7, 0.2, 0.05, 0.05]     # Lower entropy
        extreme = [0.99, 0.005, 0.003, 0.002]  # Very low entropy
        
        distributions = [
            ("Uniform", uniform),
            ("Skewed", skewed),
            ("Extreme", extreme)
        ]
        
        print("Entropy Analysis:")
        print("-" * 20)
        for name, dist in distributions:
            entropy = InformationTheory.entropy_calculation(dist)
            print(f"{name:8}: {dist} → H = {entropy:.3f} bits")
        
        print(f"\nMaximum possible entropy (uniform): {np.log2(4):.3f} bits")
        
        # KL Divergence examples
        print("\nKL Divergence Analysis:")
        print("-" * 25)
        
        p = [0.5, 0.3, 0.2]  # True distribution
        q1 = [0.5, 0.3, 0.2]  # Same as p
        q2 = [0.33, 0.33, 0.34]  # Uniform approximation
        q3 = [0.8, 0.1, 0.1]  # Poor approximation
        
        kl1 = InformationTheory.kl_divergence(p, q1)
        kl2 = InformationTheory.kl_divergence(p, q2)
        kl3 = InformationTheory.kl_divergence(p, q3)
        
        print(f"D_KL(P||P):     {kl1:.6f} (identical distributions)")
        print(f"D_KL(P||Q_uniform): {kl2:.6f} (moderate difference)")
        print(f"D_KL(P||Q_poor):    {kl3:.6f} (large difference)")
        
        return entropy, kl1, kl2, kl3

# Practical ML Applications
class MLProbabilityApplications:
    """
    Connect probability concepts to machine learning applications
    """
    
    @staticmethod
    def naive_bayes_classifier(X_train, y_train, X_test):
        """
        Simple Naive Bayes implementation using probability concepts
        """
        print("Naive Bayes Classifier Implementation")
        print("=" * 40)
        
        # Calculate class priors
        classes = np.unique(y_train)
        class_priors = {}
        for c in classes:
            class_priors[c] = np.mean(y_train == c)
            print(f"P(Class={c}) = {class_priors[c]:.3f}")
        
        # Calculate feature likelihoods (assuming Gaussian)
        feature_stats = {}
        for c in classes:
            mask = y_train == c
            feature_stats[c] = {
                'mean': np.mean(X_train[mask], axis=0),
                'std': np.std(X_train[mask], axis=0)
            }
        
        # Make predictions
        predictions = []
        for x in X_test:
            class_posteriors = {}
            for c in classes:
                # Calculate log-likelihood to avoid numerical issues
                log_likelihood = np.sum(stats.norm.logpdf(
                    x, feature_stats[c]['mean'], feature_stats[c]['std']
                ))
                log_prior = np.log(class_priors[c])
                class_posteriors[c] = log_likelihood + log_prior
            
            # Predict class with highest posterior
            predicted_class = max(class_posteriors, key=class_posteriors.get)
            predictions.append(predicted_class)
        
        return np.array(predictions)
    
    @staticmethod
    def confidence_intervals_demo():
        """
        Demonstrate confidence intervals for model evaluation
        """
        print("Confidence Intervals for Model Performance")
        print("=" * 45)
        
        # Simulate model accuracies from cross-validation
        np.random.seed(42)
        cv_accuracies = np.random.normal(0.85, 0.03, 10)  # 10-fold CV
        
        mean_acc = np.mean(cv_accuracies)
        std_acc = np.std(cv_accuracies, ddof=1)
        n = len(cv_accuracies)
        
        # Calculate 95% confidence interval
        alpha = 0.05
        t_critical = stats.t.ppf(1 - alpha/2, n-1)
        margin_error = t_critical * std_acc / np.sqrt(n)
        
        ci_lower = mean_acc - margin_error
        ci_upper = mean_acc + margin_error
        
        print(f"Cross-validation accuracies: {cv_accuracies}")
        print(f"Mean accuracy: {mean_acc:.4f}")
        print(f"Standard deviation: {std_acc:.4f}")
        print(f"95% Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")
        print(f"Interpretation: We're 95% confident the true accuracy is between {ci_lower:.1%} and {ci_upper:.1%}")

# Main demonstration
if __name__ == "__main__":
    print("Probability Theory and Statistics for Machine Learning")
    print("=" * 55)
    
    # 1. Bayes' Theorem
    prob_toolkit = ProbabilityToolkit()
    posterior = prob_toolkit.bayes_theorem_demo()
    
    print("\n" + "="*55)
    
    # 2. Distribution Gallery
    prob_toolkit.distribution_gallery()
    
    # 3. Statistical Inference
    stat_inference = StatisticalInference()
    stat_inference.central_limit_theorem_demo()
    
    # Hypothesis testing with sample data
    print("\n" + "="*55)
    np.random.seed(42)
    sample_a = np.random.normal(100, 15, 30)  # Group A
    sample_b = np.random.normal(105, 15, 30)  # Group B (slightly higher mean)
    t_stat, p_val = stat_inference.hypothesis_test_demo(sample_a, sample_b)
    
    print("\n" + "="*55)
    
    # 4. Information Theory
    info_theory = InformationTheory()
    entropy, kl1, kl2, kl3 = info_theory.information_theory_demo()
    
    print("\n" + "="*55)
    
    # 5. ML Applications
    ml_apps = MLProbabilityApplications()
    
    # Generate sample data for Naive Bayes
    np.random.seed(42)
    X_train = np.random.randn(100, 2)
    y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)
    X_test = np.random.randn(10, 2)
    
    predictions = ml_apps.naive_bayes_classifier(X_train, y_train, X_test)
    print(f"Naive Bayes predictions: {predictions}")
    
    print("\n" + "-"*55)
    
    # Confidence intervals
    ml_apps.confidence_intervals_demo()
    
    print("\n" + "="*55)
    print("Key Takeaways:")
    print("- Bayes' theorem updates beliefs with evidence")
    print("- Different distributions model different types of data")
    print("- Central Limit Theorem enables statistical inference")
    print("- Hypothesis testing validates model performance")
    print("- Entropy measures information content")
    print("- KL divergence compares probability distributions")
    print("- These concepts underpin many ML algorithms!")