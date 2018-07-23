from abc import ABCMeta,abstractmethod



class BasicModel(metaclass=ABCMeta):

    def __init__(self):pass

    @abstractmethod
    def __build_model__(self):pass

    @abstractmethod
    def __train__(self):pass

    @abstractmethod
    def __infer__(self):pass

    @abstractmethod
    def __server__(self):pass


