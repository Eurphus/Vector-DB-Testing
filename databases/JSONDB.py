import json
import os

from databases.database import Database


class JSONDB(Database):
    def __init__(self,
                 filename: str = "data",
                 homedirectory: str = os.getcwd()
                 ) -> None:
        super().__init__()
        self.filename = filename
        self.home_directory = homedirectory

    def upload(self, batch: list) -> None:
        """Method to dump list data into a JSON file

        :param batch: List of documents to dump
        """
        directory_path = self.home_directory + self.filename
        with open(directory_path, "w") as file:
            json.dump(batch, file)
