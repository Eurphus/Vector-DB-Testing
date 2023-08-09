Specs:
- **CPU:** 5600X w/PBO (Up to 4.6Mhz)
- **GPU:** RTX 3090 w/ Cuda 11.8
- **Ram:** 16GB 3000Mhz CL16 ram
- **Storage:** M.2 NVME SSD, unsure of IO speeds but it would not be limiting in this benchmark
Thermal throttling did not happen in any of these benchmarks. Temperatures were always within reasonable ranges.

# Pre-processing Benchmarks
[Code Commit](https://github.com/Eurphus/Vector-DB-Testing/tree/0b88a65250776860a0efa5c4af21b6e9624759c4)
Using concurrent threadpoolexecutor. Loading 100 various PDF files with different formats and lengths of text.
For GPU runs, the embedding model was redefined per every thread except for single batches. Otherwise it was defined once.

### TO PINECONE
Batches of 5 using CPU: 2137 seconds | I know something went wrong during this upload, but the idea is known.
Batches of 20 using CPU: 834 seconds

Batches of 1 using GPU: 905 seconds
Batches of 5 using GPU: 344 seconds
Batches of 20 using GPU: 243 seconds
Batches of 50 using GPU: 180 seconds | Massive RAM usage
Does more burst GPU tasks, significantly less CPU usage.

### TO LOCAL JSON (No API wait times)
Batches of 1 using GPU: 66|68 seconds
Batches of 5 using GPU: 110|110 seconds
Batches of 20 using GPU: 119|122 seconds
Batches of 50 using GPU: 107 seconds | Maxed out ram usage

Batches of 1 using CPU: 916 seconds
Batches of 5 using CPU: 825 seconds
Batches of 20 using CPU: 744 seconds

#### Analysis
I was surprised at the drastic difference between Pinecone and local. The act of uploading to pinecone alone takes a lot of time and should be done async. There is still a lot of optimization to be done. The main problem is using threads as well, processes or workers with their own embedding model will be more efficient. The lack of this lead to the increase in time as batch times increased during local writing.
The code is definetely not very optimized at this point.
