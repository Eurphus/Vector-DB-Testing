
# Key Takeaways
### Pre-processing
Llama_index is great, but it’s reliability on nodes Is a real hindrance. TextNodes seem very useful, but with it come a lot of unnecessary data that you upload with your vector which just takes up storage and performance. Due to this reason alone llama_index is not to be used for its complicating of the vector upload process. The extra data is just too much when in a production scale dataset. I also dislike how annoying it is to write code with it sometimes, so many strange rules and unnecessary steps. TextNodes can be a valuable powerful asset if everything is managed properly, with all the extra data being stored locally and not hindering retrieval speeds or memory.

LangChain makes for an easier way to interface data with fewer rules (Depends on context though). I definitely don’t think LangChain is production ready and grip tape should be used instead.
Having full control over the entire process may be preferred as well in a production environment.

Overall, I can't find something that fits easily exactly as I want it. Here is my understanding for how I think this process should go:

Documents will be loaded one by one and processed immediately. Once processed, it will be embedded and uploaded to the vector database along with all of it's associated text and metadata. The efficiency of this method is something I am not sure of, but going one by one and induce bottlenecks. I believe this entirely depends on the ability to multi-thread and process multiple documents at the same time, so that in the future delays from fetching from API's or IO speeds in particular areas will not drastically slow down the entire process. This option will solve the issue with memory eating and high requirements for uploading instances, as it only needs to load documents one by one.

Alternatively and potentially more CPU-efficient would be chunking the documents. Loading 50 PDFs at the same time, embedding and uploading them all at the same time. I am sure how the chunk sizes are determined can be configured in a way to take full efficiency of the device it is running on.
This is my current code setup, I will do benchmarks to determine what is more efficient.

### Embeddings / Transformers
Picking sentence transformers was not too difficult.

- **All-Minilm-v2** was good and fast but I was worried about scalability with massive amounts of data. Not the most precise but still good, just not for production.

- **E5-large-v2** failed basic data retrieval on a small scale in my tests. Due to this I will not be considering it on a larger scale either.

- Everything takes longer with **instructor** embeddings, but they likely will give a lot better data precision. I have not properly tested these yet.

- **all-mpnet-v2** is just all around best in terms of speed and performance, maybe not as precise as instructor embeddings but speed and performance may be preferred in a production environment.

- **Ada (OPENAI)** embeddings are going to be a lot more expensive than typical local sentence transformers. In the past, I have had issues processing massive datasets due to rate limits and other restrictions imposed by their model. Having data sent in chunks will solve this problem.
  - It is important to consider that ada embeds in a 1536 dimension as well. This WILL lead to higher semantic search times due to the extra math required for so many more dimensions. Again this would lead to higher precision as well, but other models can also do high dimensional embedding without the use of a pricey external API. 

### Vector Databases
#### Rejected Databases
- Chroma
  - Lack server deployment options. For small projects this will always be the best choice though I think.
- PGVector
  - Performs very poorly in benchmarks for some reason. May not be representative of real world performance though. See [here](https://github.com/erikbern/ann-benchmarks)
  - If PostreSQL database is already planned to be in use or it is practical for the given data, I have no doubt that PGVector would be the best option. However, I would not use the pgvector library on its own or switch to PostreSQL for it.
- Elastic
  - Slow and clunky 

#### Currently Investigating
- Pinecone
- Weaviate
- Milvus
- QDrant


## Considerations from the whole process
- PDF's have so much junk and most loaders eat that up. For stripping text from PDF's, there is a lot of garbage especially when using a reading solution from langchain or llama_index. Not having control over that process made it harder than it would have been to make a custom directory PDF loader.
  - I think almost any other format is superior to PDFs for reading from mass amounts of data. So many issues with encoding, junk characters and unreadable PDF's ruining text and jamming the pipeline.
- Text data can be stored in all vector databases if needed. This will drastically simplify the storage process. A lot of vector databases will allow this in the form of metadata, but not allowing the separate storing of text which can be a issue. Text is too large and memory intensive to also be stored in the memory, and metadata can be stored on disk but then metadata filtering won't be availiable.
  - If the vector database does not properly handle this, the vector upload process may be simplified by only uploaded the vectors and a ID. When semantic search is performed, search a local or NoSQL database from the associated ID.