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
| Batch Size | CPU  | GPU |
|------------|------|-----|
| 1          | N/A  | 905 |
| 5          | 2137 | 344 |
| 20         | 834  | 243 |
| 50         | N/A  | 180 |

Does more burst GPU tasks, significantly less CPU usage. Very high memory use on Batch Size = 50

### TO LOCAL JSON (No API wait times)

| Batch Size | CPU  | GPU       |
|------------|------|-----------|
| 1          | 916s | 66s,68s   |
| 5          | 825s | 110s,110s |
| 20         | 744s | 119s,122s |

#### Analysis
I was surprised at the drastic difference between Pinecone and local. The act of uploading to pinecone alone takes a lot of time and should be done async. There is still a lot of optimization to be done. The main problem is using threads as well, processes or workers with their own embedding model will be more efficient. The lack of this lead to the increase in time as batch times increased during local writing.
The code is definetely not very optimized at this point.
