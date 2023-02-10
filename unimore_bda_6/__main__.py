import logging
import pymongo.errors
from .log import install_general_log_handlers

install_general_log_handlers()

from .config import config
from .database import mongo_client_from_config, reviews_collection, sample_reviews_polar, sample_reviews_varied
from .analysis import NLTKSentimentAnalyzer, TensorflowCategorySentimentAnalyzer, TensorflowPolarSentimentAnalyzer
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
            slog.debug("Selected sample_func: %s", sample_func.__name__)

            for SentimentAnalyzer in [
                TensorflowPolarSentimentAnalyzer,
                TensorflowCategorySentimentAnalyzer,
                NLTKSentimentAnalyzer,
            ]:

                slog = logging.getLogger(f"{__name__}.{sample_func.__name__}.{SentimentAnalyzer.__name__}")
                slog.debug("Selected SentimentAnalyzer: %s", SentimentAnalyzer.__name__)

                for Tokenizer in [
                    PlainTokenizer,
                    LowercaseTokenizer,
                    NLTKWordTokenizer,
                    PottsTokenizer,
                    PottsTokenizerWithNegation,
                ]:

                    slog = logging.getLogger(f"{__name__}.{sample_func.__name__}.{SentimentAnalyzer.__name__}.{Tokenizer.__name__}")
                    slog.debug("Selected Tokenizer: %s", Tokenizer.__name__)

                    run_counter = 0

                    while True:

                        slog = logging.getLogger(f"{__name__}.{sample_func.__name__}.{SentimentAnalyzer.__name__}.{Tokenizer.__name__}.{run_counter}")
                        run_counter += 1
                        slog.debug("Run #%d", run_counter)

                        if run_counter >= 100:
                            slog.fatal("Exceeded 100 runs, giving up and exiting...")
                            exit(2)

                        try:
                            slog.debug("Instantiating %s with %s...", SentimentAnalyzer.__name__, Tokenizer.__name__)
                            sa = SentimentAnalyzer(tokenizer=Tokenizer())
                        except TypeError:
                            slog.warning("%s is not supported by %s, skipping run...", SentimentAnalyzer.__name__, Tokenizer.__name__)
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
