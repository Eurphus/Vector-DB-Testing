import concurrent.futures
import json
import logging
import os
import time

import numpy as np

def delete_bad_pdfs(filename: str = 'delete.txt'):
    dir_path = os.getcwd() + "\\data\\"
    with open(filename, 'r') as file:
        lines = file.readlines()
        for line in lines:
            file_path = dir_path + f"\\{line[:(len(line)) - 1]}"
            try:
                os.remove(file_path)
            except:
                continue
        file.close()
    with open(filename, 'w') as file:
        file.close()
        logging.info(f"Deleted {len(lines)} bad PDF's")


def dump_to_json(filename: str, data) -> None:
    directory_path = os.getcwd() + "\\results\\" + filename
    data = [data]
    with open(directory_path, "w") as file:
        json.dump(data, file)


def test_database(testing_prompts: list[str], database_name: str, iterations: int = 100, database=None):
    logging.info(f"Testing {database_name} database")

    vectorized_prompts = []
    for prompt in testing_prompts:
        vectorized_prompts.append(database.encode(prompt))
    initial_time = time.time()
    quality_times = []
    with open(f"./results/{database_name}-results.json", "a") as file:
        # For testing quality of results, checking manually for quality in queried data.
        for s in testing_prompts:
            starting_time = time.time()
            json.dump(database.query(s, postprocess=True), file)
            quality_times.append(time.time() - starting_time)

    # For testing speed of results, checking time taken to query data.
    start_time_iterations = time.time()
    count = 0
    for i in range(iterations):
        for s in vectorized_prompts:
            database.query(s, postprocess=False, pre_vectorized=True)
            count = count + 1
    end_time_iterations = time.time()
    all_items = []
    for i in range(iterations):
        for s in vectorized_prompts:
            all_items.append(s)

    multithreaded_times = []
    start_time_multithreaded = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
        futures = [executor.submit(api_request_time, database=database, prompt=s) for s in all_items]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            logging.debug(f"Result: {result}")
            multithreaded_times.append(result)
    end_time_multithreaded = time.time()

    print(f"DONE! {database_name}")
    sig_fig = lambda num: np.format_float_positional(num, precision=3, unique=False, fractional=False, trim='k')
    print(sig_fig(time.time()-initial_time))
    return {
        "database": database_name,
        "Average Quality Time": sig_fig(sum(quality_times) / len(quality_times)),
        "Average Iterations Time": sig_fig((end_time_iterations - start_time_iterations) / count),
        "Average Multi-threaded Time Sim": sig_fig(sum(multithreaded_times) / len(multithreaded_times)),
        "Average Multi-threaded Time Real": sig_fig(
            (end_time_multithreaded - start_time_multithreaded) / len(multithreaded_times)),
        "Total Quality Time": sig_fig(sum(quality_times)),
        "Total Iterations Time": sig_fig(end_time_iterations - start_time_iterations),
        "Time Multi-threaded Time": sig_fig(end_time_multithreaded - start_time_multithreaded),
        "Total All Time": sig_fig(time.time() - initial_time)
    }


def api_request_time(database, prompt):
    start_time = time.time()
    database.query(prompt, postprocess=False, pre_vectorized=True)
    return time.time() - start_time
