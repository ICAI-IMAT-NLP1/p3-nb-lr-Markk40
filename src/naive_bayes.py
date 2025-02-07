import torch
from collections import Counter
from typing import Dict

try:
    from src.utils import SentimentExample
    from src.data_processing import bag_of_words
except ImportError:
    from utils import SentimentExample
    from data_processing import bag_of_words


class NaiveBayes:
    def __init__(self):
        """
        Initializes the Naive Bayes classifier
        """
        self.class_priors: Dict[int, torch.Tensor] = None
        self.conditional_probabilities: Dict[int, torch.Tensor] = None
        self.vocab_size: int = None

    def fit(self, features: torch.Tensor, labels: torch.Tensor, delta: float = 1.0):
        """
        Trains the Naive Bayes classifier by initializing class priors and estimating conditional probabilities.

        Args:
            features (torch.Tensor): Bag of words representations of the training examples.
            labels (torch.Tensor): Labels corresponding to each training example.
            delta (float): Smoothing parameter for Laplace smoothing.
        """
        self.class_priors = self.estimate_class_priors(labels)
        self.vocab_size = len(features[0]) # Shape of the probability tensors, useful for predictions and conditional probabilities
        self.conditional_probabilities = self.estimate_conditional_probabilities(features, labels, delta)
        return

    def estimate_class_priors(self, labels: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        Estimates class prior probabilities from the given labels.

        Args:
            labels (torch.Tensor): Labels corresponding to each training example.

        Returns:
            Dict[int, torch.Tensor]: A dictionary mapping class labels to their estimated prior probabilities.
        """
        class_priors: Dict[int, torch.Tensor] = {}
        total_samples = labels.shape[0]  # Number of training examples

        unique_classes, class_counts = torch.unique(labels, return_counts=True)
        
        for c, count in zip(unique_classes, class_counts):
            class_priors[int(c)] = count / total_samples  # Correct probability formula

        return class_priors

    def estimate_conditional_probabilities(
        self, features: torch.Tensor, labels: torch.Tensor, delta: float
    ) -> Dict[int, torch.Tensor]:
        """
        Estimates conditional probabilities of words given a class using Laplace smoothing.

        Args:
            features (torch.Tensor): Bag of words representations of the training examples.
            labels (torch.Tensor): Labels corresponding to each training example.
            delta (float): Smoothing parameter for Laplace smoothing.

        Returns:
            Dict[int, torch.Tensor]: Conditional probabilities of each word for each class.
        """
        unique_classes = torch.unique(labels)
        class_word_counts: Dict[int, torch.Tensor] = {}

        vocab_size = features.shape[1]  # Assuming features.shape = (num_samples, vocab_size)

        for c in unique_classes:
            class_mask = labels == c  # Select examples belonging to class c
            word_counts = features[class_mask].sum(dim=0)  # Sum word occurrences for class c
            total_words = word_counts.sum()  # Total words in class c

            # Apply Laplace smoothing
            class_word_counts[c.item()] = (word_counts + delta) / (total_words + delta * vocab_size)

        return class_word_counts

    def estimate_class_posteriors(
        self,
        feature: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimate the class posteriors for a given feature using the Naive Bayes logic.

        Args:
            feature (torch.Tensor): The bag of words vector for a single example.

        Returns:
            torch.Tensor: Log posterior probabilities for each class.
        """
        if self.conditional_probabilities is None or self.class_priors is None:
            raise ValueError(
                "Model must be trained before estimating class posteriors."
            )
        log_posteriors: torch.Tensor = torch.zeros(len(self.class_priors))
        # p_x = sum([self.conditional_probabilities[i]@feature * self.class_priors[i] for i in range(len(self.class_priors))])
        # for i in range(len(log_posteriors)):
        #     log_posteriors[i] = torch.log(self.conditional_probabilities[i]@feature * self.class_priors[i] / p_x)
        
        for label in self.class_priors.keys():
            log_prior = torch.log(self.class_priors[label])
            log_cond_prob = torch.sum(feature * torch.log(self.conditional_probabilities[label]))
            log_posteriors[label] = log_prior + log_cond_prob

        return log_posteriors

    def predict(self, feature: torch.Tensor) -> int:
        """
        Classifies a new feature using the trained Naive Bayes classifier.

        Args:
            feature (torch.Tensor): The feature vector (bag of words representation) of the example to classify.

        Returns:
            int: The predicted class label (0 or 1 in binary classification).

        Raises:
            Exception: If the model has not been trained before calling this method.
        """
        if not self.class_priors or not self.conditional_probabilities:
            raise Exception("Model not trained. Please call the train method first.")
        
        pred: int = torch.argmax(self.estimate_class_posteriors(feature)).item()
        return pred

    def predict_proba(self, feature: torch.Tensor) -> torch.Tensor:
        """
        Predict the probability distribution over classes for a given feature vector.

        Args:
            feature (torch.Tensor): The feature vector (bag of words representation) of the example.

        Returns:
            torch.Tensor: A tensor representing the probability distribution over all classes.

        Raises:
            Exception: If the model has not been trained before calling this method.
        """
        if not self.class_priors or not self.conditional_probabilities:
            raise Exception("Model not trained. Please call the train method first.")

        probs: torch.Tensor = torch.softmax(self.estimate_class_posteriors(feature),0)
        return probs
