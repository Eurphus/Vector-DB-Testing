# Specs
- **CPU:** 5600X w/PBO (Up to 4.6Mhz)
- **GPU:** RTX 3090 w/ Cuda 11.8
- **Ram:** 16GB 3000Mhz CL16 ram
- **Storage:** M.2 NVME SSD, unsure of IO speeds, but it would not be limiting in this benchmark

Thermal throttling did not happen in any of these benchmarks. Temperatures were always within reasonable ranges.

All tests were done on the same machine with the same software and monitoring tools.


# Querying Speed & Quality Benchmarks

## 100 Iterations, basic questions
[Git Commit]()


Done with 1000 iterations, 4 random strings. 
All strings were embedded before querying, embedding speed was therefore not a limiting factor.
Quality is not comparable in this test, all items returned the EXACT same items.
Probably not comparable to real world performance


| Database                         | QDrant  | Milvus  | Pinecone |
|----------------------------------|---------|---------|----------|
| Average Quality Time             | 0.0818  | 0.127   | 0.207    |
| Average Iterations Time          | 0.0360  | 0.0773  | 0.0903   |
| Average Multi-threaded Time Sim  | 0.0792  | 0.0842  | 0.241    |
| Average Multi-threaded Time Real | 0.00398 | 0.00422 | 0.0121   |
| Total Quality Time               | 0.327   | 0.506   | 0.828    |
| Total Iterations Time            | 144.    | 309.    | 361.     |
| Time Multi-threaded Time         | 15.9    | 16.9    | 48.4     |
| Total All Time                   | 160     | 326     | 410      |

From this one graph alone it is easy to distinguish QDrant has the absolute clear winner, especially as quality was the same across all tests.
However, I would still say Milvus is still in the race.
Pinecone is a goner for sure though. It likely does not have as good computing power on the free tier then the others, but it lagged behind really bad here.
Looking at the tests, other than iteration time, QDrant is only slightly better than Milvus in all areas. I am removing Pinecone from this race, ~~but I am going to assess how a better optimized Milvus will compete.~~
Managed Milvus is auto optimized, and anything I did manually negatively impacted performance at this scale. Scaling up DB and testing.


# Pre-processing Benchmarks 
## Pre-processing Benchmark V3
[Code Commit](https://github.com/Eurphus/Vector-DB-Testing/tree/678bf7b28bd9ad0ebdae15ae3ada29743209b894)

Loading 50 various PDF files with different formats and lengths of text to PINECONE database


### To PINECONE with 50 files
200 batch size

|  Workers           | CPU | GPU |
|--------------------|-----|-----|
| 1 (non-concurrent) | 600 | 51  |
| 1                  | 580 | 40  |
| 2                  | 380 | 41  |
| 4                  | 399 | 43  |
| 12                 | 394 | 42  |
| 16                 | 392 | 42  |
| 32                 | 390 | 46  |

100 files, 200 batch size
|  Workers | GPU |
|----------|-----|
| 1        | 56  |
| 2        | 60  |
| 4        | 68  |
| 12       | 73  |

100 files
| Batch Size | 1 GPU Worker | 2 GPU Worker |
|------------|--------------|--------------|
| 400        | 56           | 60           |
| 200        | 56           | 61           |
| 100        | 56           | 62           |
| 50         | 56           | 64           |



#### Analysis
Non-multithreading loads were always longer, even with GPU. Interestingly, GPU running with a single concurrent worker was the absolute fastest for some reason. I do not fully understand why, but it may relate to my next point.
The optimal number of workers is dependent on the size of your data set. The larger the data, the better more workers are. However, the more workers the slower it is for very large PDFs to get encoded. Heavy PDFs take very long, and with loads distributed amongst so many threads, it takes even longer. This leads to higher # of workers completing the first 40 PDFs very fast, but struggling more on the last few as the machine is not focusing as much as it should on them. 

For GPU, I would not recommend going over 12 on any hardware. Diminishing returns occur even before than, and the extra Sentence Transformer loading time is not worth it. If all PDF files are the same size, testing should be determined to see the best one for both.


## Pre-processing Benchmark V2
[Code Commit](https://github.com/Eurphus/Vector-DB-Testing/tree/fc1caef87131ceae52013c7654bd080036c43261)

Loading 100 various PDF files with different formats and lengths of text.
For all runs, sentence transformers were defined at the beginning and never overlapped while running. 

### TO LOCAL JSON
| Batch Size | CPU | GPU |
|------------|-----|-----|
| 1          | 895 | 57  |
| 5          | 820 | 50s |
| 12         | 789 | 51s |
| 20         | 661 | 51s |
| 50         | N/A | 51s |

## Pre-processing Benchmarks V1
[Code Commit](https://github.com/Eurphus/Vector-DB-Testing/tree/0b88a65250776860a0efa5c4af21b6e9624759c4)

Using concurrent ThreadPoolExecutor. Loading 100 various PDF files with different formats and lengths of text.
For GPU runs, the embedding model was redefined per every thread except for single batches. Otherwise, it was defined once.

### TO PINECONE
| Batch Size | CPU  | GPU |
|------------|------|-----|
| 1          | N/A  | 905 |
| 5          | 2137 | 344 |
| 20         | 834  | 243 |
| 50         | N/A  | 180 |

Does more burst GPU tasks, significantly less CPU usage. Very high memory use on Batch Size = 50

### TO LOCAL JSON

| Batch Size | CPU  | GPU       |
|------------|------|-----------|
| 1          | 916s | 66s,68s   |
| 5          | 825s | 110s,110s |
| 20         | 744s | 119s,122s |

#### Analysis
I was surprised at the drastic difference between Pinecone and local. The act of uploading to pinecone alone takes a lot of time and should be done async. There is still a lot of optimization to be done. The main problem is using threads as well, processes or workers with their own embedding model will be more efficient. The lack of this lead to the increase in time as batch times increased during local writing.
The code is definitely not very optimized at this point.
