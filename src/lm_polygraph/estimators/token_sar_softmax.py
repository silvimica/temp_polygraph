import numpy as np

from typing import Dict

from .estimator import Estimator


class TokenSAR_softmax(Estimator):
    """
    Estimates the sequence-level uncertainty of a language model following the method of
    "Token SAR" as provided in the paper https://arxiv.org/abs/2307.01379.
    Works only with whitebox models (initialized using lm_polygraph.utils.model.WhiteboxModel).

    This method calculates the weighted sum of log_likelihoods with weights computed using token relevance.
    """

    def __init__(self, verbose: bool = False, temperature = 0.01):
        super().__init__(["token_similarity", "greedy_log_likelihoods"], "sequence")
        self.verbose = verbose
        self.temperature = temperature

    def __str__(self):
        # Return adaptive class name with temperature
        return f"TokenSAR_softmax{self.temperature}"

    def softmax(self, x: np.ndarray) -> np.ndarray:
        """
        Applies the softmax function to an array.

        Parameters:
            x (np.ndarray): Input array for which softmax will be computed.
        Returns:
            np.ndarray: Softmax-transformed probabilities.
        """
        exp_x = np.exp(x - np.max(x))  # Subtract max to prevent overflow
        return exp_x / exp_x.sum()

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Estimates the tokenSAR for each sample in the input statistics.

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics, which for multiple samples includes:
                * log p(y_i | y_<i, x) in 'greedy_log_likelihoods'
                * similarity of the generated text and generated text without one token for each token in 'token_similarity',
        Returns:
            np.ndarray: float tokenSAR for each sample in input statistics.
                Higher values indicate more uncertain samples.
        """
        batch_log_likelihoods = stats["greedy_log_likelihoods"]
        batch_token_similarity = stats["token_similarity"]

        tokenSAR = []
        for log_likelihoods, token_similarity in zip(
            batch_log_likelihoods, batch_token_similarity
        ):
            log_likelihoods = np.array(log_likelihoods) / self.temperature  # Apply temperature scaling
            log_likelihoods = self.softmax(log_likelihoods)  # Apply softmax to scaled log-likelihoods

            R_t = 1 - token_similarity
            E_t = -log_likelihoods * R_t
            
            tokenSAR.append(E_t.sum())

        return np.array(tokenSAR)