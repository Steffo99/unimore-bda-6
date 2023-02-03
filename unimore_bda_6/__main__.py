import logging

from .config import config, DATA_SET_SIZE
from .database import mongo_reviews_collection_from_config, polar_dataset, varied_dataset
from .analysis.nltk_sentiment import NLTKSentimentAnalyzer
from .tokenizer import NLTKWordTokenizer, PottsTokenizer, PottsTokenizerWithNegation
from .log import install_log_handler

log = logging.getLogger(__name__)


def main():
    for dataset_func in [polar_dataset, varied_dataset]:
        for SentimentAnalyzer in [NLTKSentimentAnalyzer]:
            for Tokenizer in [NLTKWordTokenizer, PottsTokenizer, PottsTokenizerWithNegation]:
                tokenizer = Tokenizer()
                model = SentimentAnalyzer(tokenizer=tokenizer)

                with mongo_reviews_collection_from_config() as reviews:
                    reviews_training = dataset_func(collection=reviews, amount=DATA_SET_SIZE.__wrapped__)
                    reviews_evaluation = dataset_func(collection=reviews, amount=DATA_SET_SIZE.__wrapped__)

                    log.info("Training model %s", model)
                    model.train(reviews_training)
                    log.info("Evaluating model %s", model)
                    correct, evaluated = model.evaluate(reviews_evaluation)
                    log.info("%d evaluated, %d correct, %0.2d %% accuracy", evaluated, correct, correct / evaluated * 100)

                # try:
                #     print("Manual testing for %s" % model)
                #     print("Input an empty string to continue to the next model.")
                #     while inp := input():
                #         print(model.use(inp))
                # except KeyboardInterrupt:
                #     pass


if __name__ == "__main__":
    install_log_handler()
    config.proxies.resolve()
    main()
