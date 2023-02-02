import logging

from .config import config
from .database import mongo_reviews_collection_from_config, get_training_reviews, get_test_reviews
from .analysis.vanilla import VanillaReviewSA
from .analysis.potts import PottsReviewSA
from .log import install_log_handler

log = logging.getLogger(__name__)


def main():
    with mongo_reviews_collection_from_config() as reviews:
        training_reviews = get_training_reviews(collection=reviews)
        test_reviews = get_test_reviews(collection=reviews)

    vanilla = VanillaReviewSA()
    vanilla.train(training_reviews)
    log.info("Vanilla evaluation results: %s", vanilla.evaluate(test_reviews))

    potts = PottsReviewSA()
    potts.train(training_reviews)
    log.info("Potts evaluation results: %s", potts.evaluate(test_reviews))


if __name__ == "__main__":
    install_log_handler()
    config.proxies.resolve()
    main()
