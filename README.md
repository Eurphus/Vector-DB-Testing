[See benchmarks](https://github.com/Eurphus/Vector-DB-Testing/blob/master/Benchmarks.md)

# How to use
- Ensure appropriate packages are installed. Make sure the following are installed: sentence_transformers tqdm pandas PyPDF2 torch asyncio concurrent re qdrant_client pinecone pymilvus dotenv
- Upload all PDF's into /data/, or other directory if specified.
  - If you have a metadata file for more information regarding PDF files, upload it to the home directory and name it metadata.csv or change the name. Code likely will need to be changed in get_metadata() in MacLoader in order to accommodate your specific metadata file.
- Input connection details in .env. See .env-EXAMPLE for what your .env should look like.
  - This can be avoided if you just want to input the details as a part of the DB initialization in the code instead.
- Create an index python file. From here I would define appropriate logging. With all the data it's important to look out for any issues. All logging is appropriately named using info, warning, error & critical.
- Create delete.txt, do not put anything in it.
- Import appropriate files from the project and initialize them with the appropriate details. All available params are well documented and should be shown by your IDE via hovering or clicking.
  - Ex: `from Loader import MacLoader` & `from databases.PineconeDB import PineconeDB`
- After doing a run of your files, check out delete.txt and see which files are giving issues with the PyPDF2 reader. If you want to remove these files, use Utility.delete_bad_pdfs()
- Enjoy, let me know how to improve this process!

# Key Takeaways
### Pre-processing
Llama_index is great, but it’s reliability on nodes Is a real hindrance. TextNodes seem very useful, but with it come a lot of unnecessary data that you upload with your vector which just takes up storage and performance. Due to this reason alone llama_index is not to be used for its complicating of the vector upload process. The extra data is just too much when in a production scale dataset. I also dislike how annoying it is to write code with it sometimes, so many strange rules and unnecessary steps. TextNodes can be a valuable powerful asset if everything is managed properly, with all the extra data being stored locally and not hindering retrieval speeds or memory.

LangChain makes for an easier way to interface data with fewer rules (Depends on context though). I definitely don’t think LangChain is production ready and grip tape should be used instead.
Having full control over the entire process may be preferred as well in a production environment.

Overall, I can't find something that fits easily exactly as I want it. Here is my understanding for how I think this process should go:

While finishing this program, I realized it was kind of just a replacement for the two above tools. The two above tools are meant to apply to all use cases, I just coded one specific to mine.
### Embeddings / Transformers
Picking sentence transformers was not too difficult.

- **All-Minilm-v2** was good and fast but I was worried about scalability with massive amounts of data. Not the most precise but still good, just not for production.

- **E5-large-v2** failed basic data retrieval on a small scale in my tests. Due to this I will not be considering it on a larger scale either.

- Everything takes longer with **instructor** embeddings, but they likely will give a lot better data precision. I have not properly tested these yet.

- **all-mpnet-v2** is just all around best in terms of speed and performance, maybe not as precise as instructor embeddings but speed and performance may be preferred in a production environment.

- **Ada (OPENAI)** embeddings are going to be a lot more expensive than typical local sentence transformers. In the past, I have had issues processing massive datasets due to rate limits and other restrictions imposed by their model. Having data sent in chunks will solve this problem.
  - It is important to consider that ada embeds in a 1536 dimension as well. This WILL lead to higher semantic search times due to the extra math required for so many more dimensions. Again this would lead to higher precision as well, but other models can also do high dimensional embedding without the use of a pricey external API. 
  - If the use of a local Sentence Transformer is limiting or inconvenient, this is a great replacement.

**Final Choice:** Undetermined. I need to finish pre-processing first before choosing between instruct or mpnet

### Vector Databases
#### Rejected Databases
- Chroma
  - Lack server deployment options. For small projects this will always be the best choice though I think.
- PGVector
  - Performs very poorly in benchmarks for some reason. May not be representative of real world performance though. See [here](https://github.com/erikbern/ann-benchmarks)
  - If PostreSQL database is already planned to be in use, or it is practical for the given data, I have no doubt that PGVector would be the best option. However, I would not use the pgvector library on its own or switch to PostreSQL for it.
- Elastic
  - Slow and clunky 
- Weaviate
  - I had previously planned to evaluate this, but while attempting to implement this in this program I had a lot more difficulty compared to all the others. 
    - The documentation for the python client is not great at all, it wasn't crazy difficult but the lack of documentation is worrying.
    - My use case is planning to use a managed cloud, and the WCS made me really worry. There does not seem to be much security, there's not even 2FA. I don't personally trust WCS, and it had the worst UI compared to any of the other databases.
    - If planned to use in a docker container, Weaviate should be reconsidered and given a fair shot but for now I am not continuing to evaluate it.

#### Currently Investigating
- Pinecone
- Milvus
- QDrant


## Considerations from the whole process
- PDF's have so much junk and most loaders eat that up. For stripping text from PDF's, there is a lot of garbage especially when using a reading solution from langchain or llama_index. Not having control over that process made it harder than it would have been to make a custom directory PDF loader.
  - I think almost any other format is superior to PDFs for reading from mass amounts of data. So many issues with encoding, junk characters and unreadable PDF's ruining text and jamming the pipeline.
- Text data can be stored in all vector databases if needed. This will drastically simplify the storage process. A lot of vector databases will allow this in the form of metadata, but not allowing the separate storing of text which can be an issue. Text is too large and memory intensive to also be stored in the memory, and metadata can be stored on disk but then metadata filtering won't be availiable.
  - If the vector database does not properly handle this, the vector upload process may be simplified by only uploaded the vectors and an ID. When semantic search is performed, search a local or NoSQL database from the associated ID.
