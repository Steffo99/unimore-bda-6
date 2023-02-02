import logging

from .config import config, DATA_SET_SIZE
from .database import Review, mongo_reviews_collection_from_config, dataset_polar, dataset_varied
from .analysis.vanilla import VanillaSA
from .tokenization import all_tokenizers
from .log import install_log_handler

log = logging.getLogger(__name__)


def review_vanilla_extractor(review: Review) -> tuple[str, float]:
    """
    Extract review text and rating from a `Review`.
    """
    return review["reviewText"], review["overall"]


def polar_categorizer(rating: float) -> str:
    """
    Return the polar label corresponding to the given rating.

    Possible categories are:

    * negative (1.0, 2.0)
    * positive (3.0, 4.0, 5.0)
    * unknown (everything else)
    """
    match rating:
        case 1.0 | 2.0:
            return "negative"
        case 3.0 | 4.0 | 5.0:
            return "positive"
        case _:
            return "unknown"


def varied_categorizer(rating: float) -> str:
    """
    Return the "stars" label corresponding to the given rating.

    Possible categories are:

    * terrible (1.0)
    * negative (2.0)
    * mixed (3.0)
    * positive (4.0)
    * great (5.0)
    * unknown (everything else)
    """
    match rating:
        case 1.0:
            return "terrible"
        case 2.0:
            return "negative"
        case 3.0:
            return "mixed"
        case 4.0:
            return "positive"
        case 5.0:
            return "great"
        case _:
            return "unknown"


def main():
    with mongo_reviews_collection_from_config() as reviews:
        reviews_polar_training = dataset_polar(collection=reviews, amount=DATA_SET_SIZE.__wrapped__)
        reviews_polar_evaluation = dataset_polar(collection=reviews, amount=DATA_SET_SIZE.__wrapped__)

    for tokenizer in all_tokenizers:
        log.info("Training polar model with %s tokenizer", tokenizer)
        model = VanillaSA(extractor=review_vanilla_extractor, tokenizer=tokenizer, categorizer=polar_categorizer)
        model.train(reviews_polar_training)
        log.info("Evaluating polar model with %s tokenizer", tokenizer)
        evaluation = model.evaluate(reviews_polar_evaluation)
        log.info("Polar model with %s results: %s", tokenizer, evaluation)

    del reviews_polar_training
    del reviews_polar_evaluation

    with mongo_reviews_collection_from_config() as reviews:
        reviews_varied_training = dataset_varied(collection=reviews, amount=DATA_SET_SIZE.__wrapped__)
        reviews_varied_evaluation = dataset_varied(collection=reviews, amount=DATA_SET_SIZE.__wrapped__)

    for tokenizer in all_tokenizers:
        log.info("Training varied model with %s tokenizer", tokenizer)
        model = VanillaSA(extractor=review_vanilla_extractor, tokenizer=tokenizer, categorizer=varied_categorizer)
        model.train(reviews_varied_training)
        log.info("Evaluating varied model with %s tokenizer", tokenizer)
        evaluation =  model.evaluate(reviews_varied_evaluation)
        log.info("Varied model with %s results: %s", tokenizer, evaluation)

    del reviews_varied_training
    del reviews_varied_evaluation


if __name__ == "__main__":
    install_log_handler()
    config.proxies.resolve()
    main()
