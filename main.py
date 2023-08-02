import time

from PineconeDB import create_pineconedb
from QdrantDB import create_QdrantDB

starting_time = time.time()
create_pineconedb()
print(f"\n\n\nPinecone:  {time.time()-starting_time}\n\n\n")

starting_time = time.time()
create_QdrantDB()
print(f"\n\n\nQdrant:  {time.time()-starting_time}\n\n\n")