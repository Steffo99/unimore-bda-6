from .config import config
from .database import mongo_reviews_collection_from_config, get_training_reviews, get_test_reviews
from .analysis.vanilla import VanillaReviewSA
from .log import install_log_handler


def main():
    with mongo_reviews_collection_from_config() as reviews:
        training_reviews = get_training_reviews(collection=reviews)
        test_reviews = get_test_reviews(collection=reviews)

    model = VanillaReviewSA()
    model.train(training_reviews)
    
    evaluation = model.evaluate(test_reviews)
    print(evaluation)
    
    while True:
        classification = model.use(input())
        print(classification)


if __name__ == "__main__":
    install_log_handler()
    config.proxies.resolve()
    main()
