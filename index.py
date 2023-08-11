import datetime
import logging
import os

from Loader import MacLoader
from databases.PineconeDB import PineconeDB

logging_folder = os.getcwd() + "/logs/"
date_F = datetime.datetime.now().strftime("%d.%b_%Y_%H.%M.%S")

logging.basicConfig(
    level=logging.INFO,
    filename=logging_folder + f"{date_F}.txt"
)
database = PineconeDB(
    ensure_exists=True
)

loader = MacLoader()
loader.encode_docs(
    database=database,
    max_num_files=100,
    batch_size=5
)
