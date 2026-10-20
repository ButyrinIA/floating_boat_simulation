from abc import ABC, abstractmethod

import numpy as np


class BaseForce(ABC):
    @abstractmethod
    def compute(self, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        """Точка приложения и вектор силы."""
        ...
