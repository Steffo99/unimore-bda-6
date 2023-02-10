import logging
import pymongo.errors
from .log import install_log_handler

install_log_handler()

from .config import config
from .database import mongo_client_from_config, reviews_collection, sample_reviews_polar, sample_reviews_varied
from .analysis.nltk_sentiment import NLTKSentimentAnalyzer
from .analysis.tf_text import TensorflowSentimentAnalyzer
from .analysis.base import TrainingFailedError
from .tokenizer import PlainTokenizer, LowercaseTokenizer, NLTKWordTokenizer, PottsTokenizer, PottsTokenizerWithNegation
from .gathering import Caches

log = logging.getLogger(__name__)


def main():
    log.info("Started unimore-bda-6 in %s mode!", "DEBUG" if __debug__ else "PRODUCTION")

    log.debug("Validating configuration...")
    config.proxies.resolve()

    log.debug("Ensuring there are no leftover caches...")
    Caches.ensure_clean()

    with mongo_client_from_config() as db:
        try:
            db.admin.command("ping")
        except pymongo.errors.ServerSelectionTimeoutError:
            log.fatal("MongoDB database is not available, exiting...")
            exit(1)

        reviews = reviews_collection(db)

        for sample_func in [sample_reviews_varied, sample_reviews_polar]:

            slog = logging.getLogger(f"{__name__}.{sample_func.__name__}")
            slog.debug(f"Selected sample_func: %s", sample_func)

            for SentimentAnalyzer in [
                TensorflowSentimentAnalyzer,
                NLTKSentimentAnalyzer
            ]:

                slog = logging.getLogger(f"{__name__}.{sample_func.__name__}.{SentimentAnalyzer.__name__}")
                slog.debug(f"Selected SentimentAnalyzer: %s", SentimentAnalyzer)

                for Tokenizer in [
                    PlainTokenizer,
                    LowercaseTokenizer,
                    NLTKWordTokenizer,
                    PottsTokenizer,
                    PottsTokenizerWithNegation,
                ]:

                    slog = logging.getLogger(f"{__name__}.{sample_func.__name__}.{SentimentAnalyzer.__name__}.{Tokenizer.__name__}")
                    slog.debug(f"Selected Tokenizer: %s", Tokenizer)

                    while True:

                        try:
                            slog.debug("Instantiating %s with %s...", SentimentAnalyzer, Tokenizer)
                            sa = SentimentAnalyzer(tokenizer=Tokenizer())
                        except TypeError:
                            slog.warning("%s is not supported by %s, skipping run...", SentimentAnalyzer, Tokenizer)
                            break

                        with Caches.from_database_samples(collection=reviews, sample_func=sample_func) as datasets:
                            try:
                                slog.info("Training sentiment analyzer: %s", sa)
                                sa.train(training_dataset_func=datasets.training, validation_dataset_func=datasets.validation)

                            except TrainingFailedError:
                                slog.error("Training failed, trying again with a different dataset...")
                                continue

                            else:
                                slog.info("Training succeeded!")

                                slog.info("Evaluating sentiment analyzer: %s", sa)
                                evaluation_results = sa.evaluate(evaluation_dataset_func=datasets.evaluation)
                                slog.info("Evaluation results: %s", evaluation_results)
                                break


if __name__ == "__main__":
    main()
