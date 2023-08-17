from copy import deepcopy
from typing import Optional

from databases.MilvusDB import MilvusDB
from databases.PineconeDB import PineconeDB
from databases.QDrantDB import QDrantDB
from databases.database import Database


class AllDB(Database):
    """Class for uploading text to every database
    """

    def __init__(self,
                 databases: Optional[list] = None
                 ) -> None:
        super().__init__()
        if databases is None:
            databases = [MilvusDB(), PineconeDB(), QDrantDB()]
        self.databases = databases

    async def upload(self, batch):
        for db in self.databases:
            temp_batch = deepcopy(batch)
            await db.upload(temp_batch)

    def clear(self):
        for db in self.databases:
            db.clear()
