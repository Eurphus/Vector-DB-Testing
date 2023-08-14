import torch
from sentence_transformers import SentenceTransformer


class Database:
    """Abstract class for other classes to use.

    """

    def __init__(self,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 model_name: str = "all-mpnet-base-v2"
                 ) -> None:
        self.device = device
        self.model_name = model_name
        self.encoder = SentenceTransformer(model_name_or_path=model_name, device=device)
        self.model_dimensions = self.encoder.get_sentence_embedding_dimension()

    def query(self, text: str):
        """Method to query the database for data
        """
        pass

    def upload(self, batch: list) -> None:
        pass

    def clear(self, *args) -> None:
        pass

    def create(self, *args) -> None:
        pass

    def encode(self, text: str) -> list:
        return self.encoder.encode(text, show_progress_bar=False).tolist()

    def indexing(self, enable: bool) -> None:
        pass
