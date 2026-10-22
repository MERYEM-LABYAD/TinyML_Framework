from abc import ABC, abstractmethod


class QuantizationBackend(ABC):

    @abstractmethod
    def prepare(self, model, example_inputs):
        raise NotImplementedError

    @abstractmethod
    def convert(self, model):
        raise NotImplementedError