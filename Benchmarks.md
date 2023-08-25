# Specs
- **CPU:** 5600X w/PBO (Up to 4.6Mhz)
- **GPU:** RTX 3090 w/ Cuda 11.8
- **Ram:** 16GB 3000Mhz CL16 ram
- **Storage:** M.2 NVME SSD, unsure of IO speeds, but it would not be limiting in this benchmark

Thermal throttling did not happen in any of these benchmarks. Temperatures were always within reasonable ranges.

All tests were done on the same machine with the same software and monitoring tools.


# Querying Speed & Quality Benchmarks

The Milvus instance has 1 CU, which I don't know is comparable to a vCore. 

QDrant had 0.5 vCore, max 1 gb

Ultimately, Milvus hardware was billed at $65/month for a trial. 

QDrant is on a starter instance that is forever free. The two WILL have differences in speed. 

Same goes for Pinecone, but at lower vector counts it still was much slower than the other two. I am saying this for QDrant and Milvus because in lower vector count benchmarks, QDrant outperformed Milvus slightly and only in multithreading high query tests.


[Testing Method](https://github.com/Eurphus/Vector-DB-Testing/blob/master/Utility.py)

It was a very simple method for testing these stats, and is not representative of real world performance. 

Something notable is that Multi-threaded sim is the average time it takes for a individual thread to process a query. Real is the time it took for the query process to finish. Does not take any single query into consideration, just how quickly all of them finish.

### 50000 Iterations, 8 prompts, 100 workers, 178800 vectors

|                                  | Milvus       | QDrant       | Pinecone     |
|----------------------------------|--------------|--------------|--------------|
| Average Quality Time             | 0.0978       | 0.0647       | 0.144        |
| Average Multi-threaded Time Sim  | 0.148        | 0.578        | 0.823        |
| Average Multi-threaded Time Real | 0.00148      | 0.00579      | 0.00823      |
| Total Quality Time               | 0.783        | 0.518        | 1.15         |
| Time Multi-threaded Time         | 593.         | 2310.        | 3290.        |



### 1000 Iterations, 8 prompts, 100 workers, 178800 vectors

| database                         | Milvus | QDrant       | Pinecone |
|----------------------------------|--------|--------------|----------|
| Average Quality Time             | 0.0828 | 0.0499       | 0.147    |
| Average Multi-threaded Time Sim  | 0.149  | 0.543        | 0.831    |
| Average Multi-threaded Time Real | 0.0015 | 0.00547      | 0.00838  |
| Total Quality Time               | 0.663  | 0.399        | 1.17     |
| Time Multi-threaded Time         | 12.0   | 43.8         | 67.0     |


### 10000 Iterations, 8 prompts, 1000 workers, 178800 vectors


| database                         | Milvus       | QDrant       | Pinecone |
|----------------------------------|--------------|--------------|----------|
| Average Quality Time             | 0.108        | 0.0511       | FAIL     |
| Average Multi-threaded Time Sim  | 1.68         | 5.52         | FAIL     |
| Average Multi-threaded Time Real | 0.00169      | 0.00556      | FAIL     |
| Total Quality Time               | 0.862        | 0.409        | FAIL     |
| Time Multi-threaded Time         | 136.         | 445.         | FAIL     |
| Total All Time                   | 136.         | 445.         | FAIL     |

Pinecone failed due to too many GRPC requests, not sure if this was a Pinecone issue or a local package issue.
A repeat of this test at 10000 iterations gave the same results. I am sure 1000 workers was an issue

## 100 Iterations, basic questions, low vector count (>2000?)
[Git Commit](https://github.com/Eurphus/Vector-DB-Testing/tree/f41518c33d886a8c077d44f8ebb348bb0f90714f)

*Note: Done on a work laptop NOT the machine mentioned above in specs


Done with 1000 iterations, 4 random strings. With 365mb pdf base as mentioned before
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


### Pre-processing info

After attempting to upload 3GB+ pdf base with 300 batch size, 1 worker.
- The databases were being hammered the entire time, but every single database had NO FAILURES. 
- Each DB managed the load fine, with some understandable uploading spike times. 
  - Pinecone was the most spikey, with times normally ranging from 1.2 - 1.6s with occasional spikes of 2s-3s
  - QDrant was consistently the best with batch times being 0.8 - 1.4. Max spikes of 1.9 which were rare
  - Milvus was consistently the slowest with batch times beings 1.4 - 1.7 with occasional spikes of 2-2.8s during heavy load
    - I did not disable indexing, which may explain this.

I stopped the program at 46% (732 PDFs) due to Pinecone filling up first. Milvus and QDrant still have plenty of space with their starter tiers.
178800 reported vectors

## Pre-processing Benchmark V4
[Git Commit](https://github.com/Eurphus/Vector-DB-Testing/tree/f41518c33d886a8c077d44f8ebb348bb0f90714f)
This test is because I forgot my previous numbers, but wanted an estimate for how long uploading will take.
Done with multithreading, 1 concurrent workers, gpu, and a batch size of 200.
Uploading would go faster, but pinecone will not safely take a larger size.
With 75 files worth 178MB

|                             | Milvus | QDrant | Pinecone |
|-----------------------------|--------|--------|----------|
| Time to upload batch of 200 | ~1.1s  | ~0.6   | ~0.9s    |
| Time to upload all          | 101s   | 106.6  | 105.7s   |

Between the different databases this isn't a significant difference. However, Pinecone is a limiting factor due to the requirement of a lower batch size.
Upload time should not be a factor while comparing these DB's, all final times are within range of error.
Also, strange that average time to upload batches of 200 was 0.6 for QDrant, but Milvus was almost two times slower and still completed in less time.

## Pre-processing Benchmark V3
[Code Commit](https://github.com/Eurphus/Vector-DB-Testing/tree/678bf7b28bd9ad0ebdae15ae3ada29743209b894)
a
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
