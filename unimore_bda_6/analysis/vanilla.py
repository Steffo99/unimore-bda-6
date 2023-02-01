import nltk
import nltk.classify
import nltk.sentiment
import nltk.sentiment.util
import logging

from ..database import mongo_reviews_collection_from_config, get_reviews_training_set, get_reviews_test_set


log = logging.getLogger(__name__)


def create_model_vanilla() -> nltk.sentiment.SentimentAnalyzer:
    log.debug("Creating model...")
    model = nltk.sentiment.SentimentAnalyzer()
    log.debug("Created model %s!", model)
    return model


def train_model_vanilla(model: nltk.sentiment.SentimentAnalyzer) -> None:
    # TODO: This doesn't work yet

    with mongo_reviews_collection_from_config() as reviews:
        training_set = get_reviews_training_set(reviews)

    log.debug("Marking negations...")
    training_negated_set = list(map(nltk.sentiment.util.mark_negation, training_set))

    log.debug("Extracting tokens...")
    training_tokens = model.all_words(training_negated_set, labeled=False)

    log.debug("Counting unigrams...")
    training_unigrams = model.unigram_word_feats(words=training_tokens, min_freq=4)

    log.debug("Configuring model features...")
    model.add_feat_extractor(nltk.sentiment.util.extract_unigram_feats, unigrams=training_unigrams)
    training_set = model.apply_features(documents=training_set)

    log.info("Training model...")
    model.train(trainer=nltk.classify.NaiveBayesClassifier.train, training_set=training_set)


def evaluate_model_vanilla(model: nltk.sentiment.SentimentAnalyzer):
    with mongo_reviews_collection_from_config() as reviews:
        test_set = get_reviews_test_set(reviews)

    log.info("Evaluating model...")
    model.evaluate(test_set)

    # TODO
    breakpoint()


__all__ = (
    "create_model_vanilla",
    "train_model_vanilla",
    "evaluate_model_vanilla",
)
