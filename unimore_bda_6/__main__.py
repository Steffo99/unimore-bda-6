import logging
import pymongo.errors
import gc
from .log import install_general_log_handlers

install_general_log_handlers()

from .config import config, TARGET_RUNS, MAXIMUM_RUNS
from .database import mongo_client_from_config, reviews_collection, sample_reviews_polar, sample_reviews_varied
from .analysis import NLTKSentimentAnalyzer, TensorflowCategorySentimentAnalyzer, TensorflowPolarSentimentAnalyzer, ThreeCheat
from .analysis.base import TrainingFailedError, EvaluationResults
from .tokenizer import PlainTokenizer, LowercaseTokenizer, NLTKWordTokenizer, PottsTokenizer, PottsTokenizerWithNegation, HuggingBertTokenizer
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

    for sample_func in [
        sample_reviews_polar,
        sample_reviews_varied,
    ]:

        slog = logging.getLogger(f"{__name__}.{sample_func.__name__}")
        slog.debug("Selected sample_func: %s", sample_func.__name__)

        for SentimentAnalyzer in [
            # ThreeCheat,
            NLTKSentimentAnalyzer,
            TensorflowPolarSentimentAnalyzer,
            TensorflowCategorySentimentAnalyzer,
        ]:

            slog = logging.getLogger(f"{__name__}.{sample_func.__name__}.{SentimentAnalyzer.__name__}")
            slog.debug("Selected SentimentAnalyzer: %s", SentimentAnalyzer.__name__)

            for Tokenizer in [
                PlainTokenizer,
                LowercaseTokenizer,
                NLTKWordTokenizer,
                PottsTokenizer,
                PottsTokenizerWithNegation,
                HuggingBertTokenizer,
            ]:

                log.debug("Running garbage collection...")
                garbage_count = gc.collect()
                log.debug("Collected %d pieces of garbage!", garbage_count)

                slog = logging.getLogger(f"{__name__}.{sample_func.__name__}.{SentimentAnalyzer.__name__}.{Tokenizer.__name__}")
                slog.debug("Selected Tokenizer: %s", Tokenizer.__name__)

                runs = 0
                successful_runs = 0
                cumulative_evaluation_results = EvaluationResults()

                while True:

                    slog = logging.getLogger(f"{__name__}.{sample_func.__name__}.{SentimentAnalyzer.__name__}.{Tokenizer.__name__}")

                    if successful_runs >= TARGET_RUNS.__wrapped__:
                        slog.info("Reached target of %d runs, moving on...", TARGET_RUNS.__wrapped__)
                        break

                    if runs >= MAXIMUM_RUNS.__wrapped__:
                        slog.fatal("Exceeded %d runs, giving up and exiting...", MAXIMUM_RUNS.__wrapped__)
                        break

                    runs += 1
                    slog = logging.getLogger(f"{__name__}.{sample_func.__name__}.{SentimentAnalyzer.__name__}.{Tokenizer.__name__}.{runs}")
                    slog.debug("Run #%d", runs)

                    try:
                        slog.debug("Instantiating %s with %s...", SentimentAnalyzer.__name__, Tokenizer.__name__)
                        sa = SentimentAnalyzer(tokenizer=Tokenizer())
                    except TypeError:
                        slog.warning("%s is not supported by %s, skipping run...", SentimentAnalyzer.__name__, Tokenizer.__name__)
                        break


                    with mongo_client_from_config() as db:
                        reviews = reviews_collection(db)
                        datasets_cm = Caches.from_database_samples(collection=reviews, sample_func=sample_func)
                        datasets = datasets_cm.__enter__()

                    try:
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
                            successful_runs += 1
                            cumulative_evaluation_results += evaluation_results
                            break
                    finally:
                        datasets_cm.__exit__()

                slog.info("Cumulative evaluation results: %s", cumulative_evaluation_results)


if __name__ == "__main__":
    main()
