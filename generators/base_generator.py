from abc import abstractmethod
from abc import ABC

class Generator():
    @abstractmethod
    def check_configs(path):
        AssertionError()

    @abstractmethod
    def load_configs(path):
        AssertionError()

    @abstractmethod
    def load_model(model_json, img_json, tokenizer_json):
        AssertionError()

    @abstractmethod
    def get_trainer(path):
        AssertionError()
        