import logging
import tensorflow

from .config import config, DATA_SET_SIZE
from .database import mongo_client_from_config, reviews_collection, sample_reviews_polar, sample_reviews_varied, store_cache, load_cache, delete_cache
from .analysis.nltk_sentiment import NLTKSentimentAnalyzer
from .analysis.tf_text import TensorflowSentimentAnalyzer
from .analysis.base import TrainingFailedError
from .tokenizer import LowercaseTokenizer
from .log import install_log_handler

log = logging.getLogger(__name__)


def main():
    if len(tensorflow.config.list_physical_devices(device_type="GPU")) == 0:
        log.warning("Tensorflow reports no GPU acceleration available.")
    else:
        log.debug("Tensorflow successfully found GPU acceleration!")

    try:
        delete_cache("./data/training")
        delete_cache("./data/evaluation")
    except FileNotFoundError:
        pass

    for dataset_func in [sample_reviews_polar, sample_reviews_varied]:
        for SentimentAnalyzer in [TensorflowSentimentAnalyzer, NLTKSentimentAnalyzer]:
            for Tokenizer in [
                # NLTKWordTokenizer,
                # PottsTokenizer,
                # PottsTokenizerWithNegation,
                LowercaseTokenizer,
            ]:
                while True:
                    try:
                        tokenizer = Tokenizer()
                        model = SentimentAnalyzer(tokenizer=tokenizer)

                        with mongo_client_from_config() as db:
                            log.debug("Finding the reviews MongoDB collection...")
                            collection = reviews_collection(db)

                            try:
                                training_cache = load_cache("./data/training")
                                evaluation_cache = load_cache("./data/evaluation")
                            except FileNotFoundError:
                                log.debug("Gathering datasets...")
                                reviews_training = dataset_func(collection=collection, amount=DATA_SET_SIZE.__wrapped__)
                                reviews_evaluation = dataset_func(collection=collection, amount=DATA_SET_SIZE.__wrapped__)

                                log.debug("Caching datasets...")
                                store_cache(reviews_training, "./data/training")
                                store_cache(reviews_evaluation, "./data/evaluation")
                                del reviews_training
                                del reviews_evaluation

                                training_cache = load_cache("./data/training")
                                evaluation_cache = load_cache("./data/evaluation")
                                log.debug("Caches stored and loaded successfully!")
                            else:
                                log.debug("Caches loaded successfully!")

                            log.info("Training model: %s", model)
                            model.train(training_cache)
                            log.info("Evaluating model: %s", model)
                            evaluation_results = model.evaluate(evaluation_cache)
                            log.info("%s", evaluation_results)

                    except TrainingFailedError:
                        log.error("Training failed, restarting with a different dataset.")
                        continue
                    else:
                        log.info("Training")
                        break
                    finally:
                        delete_cache("./data/training")
                        delete_cache("./data/evaluation")


if __name__ == "__main__":
    install_log_handler()
    config.proxies.resolve()
    main()
