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

    with open("./data/logs/results.tsv", "w") as file:
        file.write("function\tanalyzer\ttokenizer\trun no\tmean absolute error\tmean squared error\tperfects\trecall 1\trecall 2\trecall 3\trecall 4\trecall 5\tprecision 1\tprecision 2\tprecision 3\tprecision 4\tprecision 5\n")

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
                ThreeCheat,
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

                                file.write(f"{sample_func.__name__}\t{SentimentAnalyzer.__name__}\t{Tokenizer.__name__}\t{runs}\t\t\t\t\t\t\t\t\t\t\t\t\t\n")
                                file.flush()
                                continue

                            else:
                                slog.info("Training succeeded!")
                                slog.info("Evaluating sentiment analyzer: %s", sa)
                                evaluation_results = sa.evaluate(evaluation_dataset_func=datasets.evaluation)
                                slog.info("Evaluation results: %s", evaluation_results)

                                file.write(f"{sample_func.__name__}\t{SentimentAnalyzer.__name__}\t{Tokenizer.__name__}\t{runs}\t{evaluation_results.mean_absolute_error()}\t{evaluation_results.mean_squared_error()}\t{evaluation_results.perfect_count()}\t{evaluation_results.recall(1.0)}\t{evaluation_results.recall(2.0)}\t{evaluation_results.recall(3.0)}\t{evaluation_results.recall(4.0)}\t{evaluation_results.recall(5.0)}\t{evaluation_results.precision(1.0)}\t{evaluation_results.precision(2.0)}\t{evaluation_results.precision(3.0)}\t{evaluation_results.precision(4.0)}\t{evaluation_results.precision(5.0)}\n")
                                file.flush()

                                successful_runs += 1
                                cumulative_evaluation_results += evaluation_results
                        finally:
                            datasets_cm.__exit__(None, None, None)

                    slog.info("Cumulative evaluation results: %s", cumulative_evaluation_results)


if __name__ == "__main__":
    main()
