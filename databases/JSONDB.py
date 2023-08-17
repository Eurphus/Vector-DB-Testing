import json
import os

from databases.database import Database


class JSONDB(Database):
    """ JSON Database Class for interacting with a defined JSON file

    Args:
        filename (str): Name of JSON file
        homedirectory (str): Directory of JSON file
    """

    def __init__(self,
                 filename: str = "data",
                 home_directory: str = os.getcwd()
                 ) -> None:
        super().__init__()
        self.filename = filename
        self.home_directory = home_directory

    async def upload(self, batch: list) -> None:
        """Method to dump list data into a JSON file

        :param batch: List of documents to dump
        """
        directory_path = self.home_directory + self.filename
        with open(directory_path, "w") as file:
            json.dump(batch, file)
