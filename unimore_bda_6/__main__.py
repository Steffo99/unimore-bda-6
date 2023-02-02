import logging

from .config import config, DATA_SET_SIZE
from .database import mongo_reviews_collection_from_config, get_reviews_dataset_polar, get_reviews_dataset_uniform
from .analysis.vanilla import VanillaReviewSA, VanillaUniformReviewSA
from .analysis.potts import PottsReviewSA
from .log import install_log_handler

log = logging.getLogger(__name__)


def main():
    with mongo_reviews_collection_from_config() as reviews:
        reviews_polar_training = get_reviews_dataset_polar(collection=reviews, amount=DATA_SET_SIZE.__wrapped__)
        reviews_polar_evaluation = get_reviews_dataset_polar(collection=reviews, amount=DATA_SET_SIZE.__wrapped__)
        reviews_uniform_training = get_reviews_dataset_uniform(collection=reviews, amount=DATA_SET_SIZE.__wrapped__)
        reviews_uniform_evaluation = get_reviews_dataset_uniform(collection=reviews, amount=DATA_SET_SIZE.__wrapped__)

    vanilla_polar = VanillaReviewSA()
    vanilla_polar.train(reviews_polar_training)
    log.info("Vanilla polar evaluation results: %s", vanilla_polar.evaluate(reviews_polar_evaluation))

    potts_polar = PottsReviewSA()
    potts_polar.train(reviews_polar_training)
    log.info("Potts polar evaluation results: %s", potts_polar.evaluate(reviews_polar_evaluation))

    vanilla_uniform = VanillaUniformReviewSA()
    vanilla_uniform.train(reviews_uniform_training)
    log.info("Vanilla uniform evaluation results: %s", vanilla_polar.evaluate(reviews_polar_evaluation))

    while True:
        print(vanilla_uniform.use(input("> ")))



if __name__ == "__main__":
    install_log_handler()
    config.proxies.resolve()
    main()
