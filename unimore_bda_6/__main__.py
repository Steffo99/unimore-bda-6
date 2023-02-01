from .config import config
from .analysis.vanilla import create_model_vanilla, train_model_vanilla, evaluate_model_vanilla
from .log import install_log_handler


def main():
    model = create_model_vanilla()
    train_model_vanilla(model)
    evaluate_model_vanilla(model)


if __name__ == "__main__":
    install_log_handler()
    config.proxies.resolve()
    main()
