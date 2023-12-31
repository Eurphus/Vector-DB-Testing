import asyncio
import concurrent.futures
import gc
import logging
import os
import re
import time
from typing import Optional

import PyPDF2
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

class MacLoader:
    """Multithreading directory file loading & encoding

    Args:
        device (str): Selected device to encode. If not specified it will pick CUDA if available, cpu if not.
        model_name (str): Sentence Transformer model to use.
        max_workers_num (int): Max amount of workers/threads to run at the same time. If not specified it will default to amount of threads your CPU supports.
        max_chunk_size (int): Max amount of characters per every chunk of data. Higher for greater context, lower for less context but more specific data. If descreased, make sure k search param is increased.
        chunk_overlap (int): Amount of overlap between chunks.
        data_directory (str): Directory where all data is found.
        JSONFilename (str): Name of file to write JSON data to if applicable.
        metadata_file (str): If applicable, file where metadata information is found.
    """

    def __init__(
            self,
            device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
            model_name: str = "all-mpnet-base-v2",
            multithreading: bool = True,
            max_workers_num: int = 1,
            max_chunk_size: int = 1000,
            chunk_overlap: int = 20,
            data_directory: str = "data",
            JSONFilename: str = "data",
            metadata_file: str = "metadata.csv"
    ) -> None:
        self.device = device
        self.model_name = model_name
        self.multithreading = multithreading
        self.max_workers_num = max_workers_num
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        self.data_directory = data_directory
        self.JSONFilename = JSONFilename
        self.metadata_file = metadata_file
        self.doc_count = 0

        self.logger = logging.getLogger(__name__)
        if device == 'cuda':
            self.embeddings = []
            for i in range(max_workers_num):
                self.embeddings.append(SentenceTransformer(model_name_or_path=model_name, device=device))
        else:
            self.embeddings = SentenceTransformer(model_name_or_path=model_name, device=device)
        self.encoder = lambda num: self.embeddings if self.device == 'cpu' else self.embeddings[
            num % self.max_workers_num]

    def _clean_text(self, text: str):
        """Method to help remove garbage characters from text

        :param text: String to clean
        :return: Cleaned string
        """
        fixed = text.replace("  ", " ")
        # Recursive method to reduce junk whitespace
        if "  " in fixed:
            return self._clean_text(fixed)
        else:
            # Remove line skipping associated with PDF loading
            fixed = fixed.replace("\n", "").replace(".,", ".").replace("\t", "")

            # Remove hyphens between two words where it should not exist due to PDF loading
            fixed = re.sub(r'\b(\w+)-(\w+)\b', r'\1\2', fixed)

            # Forgot what this one removes
            fixed = re.sub(r'\(cid:\d+\)', ' ', fixed)
            if "  " in fixed:
                return self._clean_text(fixed)

            # Remove starting whitespace
            if fixed.startswith(" "):
                fixed = fixed[1:]
            return fixed

    def encode_docs(self,
                    database: any = None,
                    max_num_files: Optional[int] = None,
                    batch_size: int = 500
                    ) -> None:
        """Main method for encoding and loading docs
        :params:
            database (any): Selected database to send data to. Options: JSON, Pinecone, Milvus, QDrant
            max_num_files (int): Max number of files to load & read from.
            batch_size (int): Number of accumulated batches until an upload is triggered
        :return:
        """
        self.logger.info(f"""Using device {self.device} for embedding 
                     Using {self.max_workers_num} workers
                     Using batches of {self.max_chunk_size} with a max of {max_num_files} files
                     Model Name={self.model_name}, data directory={self.data_directory}, max chunk size={self.max_chunk_size}, overlap={self.chunk_overlap}""")
        starting_time = time.time()

        # Faster uploading for applicable databases
        database.indexing(False)

        # Get current path
        directory_path = os.getcwd() + f"/{self.data_directory}/"
        directory = os.listdir(directory_path)

        dir_length = len(directory)

        if max_num_files != 0 and dir_length > max_num_files:
            dir_length = max_num_files

        pbar = tqdm(total=dir_length, desc=f"Encoding files")

        self.logger.info(f"Encoding documents...")
        batched_files = directory[:dir_length]
        batched = []

        if self.multithreading:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers_num) as executor:
                futures = [executor.submit(self.process_pdf, filename=filename, encoder=self.encoder(i)) for i, filename
                           in enumerate(batched_files)]
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    if result is None or result is []:
                        continue
                    batched.extend(result)

                    # Loops batched and uploads for each batch size
                    while len(batched) >= batch_size:
                        tmp = batched[:batch_size]
                        batched = batched[batch_size:]
                        try:
                            asyncio.run(database.upload(tmp))
                        except Exception as e:
                            self.logger.critical(f"Failed to upload, skipping batch! See batch:\n")
                            self.logger.exception(e)
                            break
                    gc.collect()
                    pbar.update(1)
        else:
            for i, filename in enumerate(batched_files):
                result = self.process_pdf(filename=filename, encoder=self.encoder(i))
                if result is None or result is []:
                    continue
                batched.extend(result)
                while len(batched) >= batch_size:
                    tmp = batched[:batch_size]
                    batched = batched[batch_size:]
                    try:
                        asyncio.run(database.upload(tmp))
                    except Exception as e:
                        self.logger.critical(f"Failed to upload, skipping batch! See batch:\n")
                        self.logger.exception(e)
                        break
                pbar.update(1)

        try:
            asyncio.run(database.upload(batched[:batch_size]))
        except Exception as e:
            self.logger.critical(f"Failed to upload, skipping batch! See batch:\n")
            self.logger.exception(e)

        self.logger.info(f"Encoded {dir_length} documents in {time.time() - starting_time} seconds")
        pbar.close()
        database.indexing(True)
        self.doc_count = 0

    def get_file_metadata(self, filename: str) -> Optional[dict]:
        # Scan metadata.csv for associated data
        df = pd.read_csv(self.metadata_file)
        filename = filename.replace(".pdf", "").replace(self.data_directory, "").replace('\\', "")
        located = df.loc[df['digest'] == filename]
        if located.empty:
            self.logger.error(f"Could not find {filename} in metadata.csv")
            return {}
        return {
            'filename': filename,
            'size': str(located['file_size'].iloc[0]),
            # Has to be str or int for some reason, was not working before. # Size is in bytes
            'created_at': located['date'].iloc[0][:10]
        }

    def process_pdf(self, filename, encoder):
        text = ""
        directory_path = os.getcwd() + f"/{self.data_directory}/"
        file_path = directory_path + filename

        try:
            with open(file_path, "rb") as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)

                # Extract text from each page
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text()

                if len(text) == 0:
                    self.logger.warning(f"PDF with filename {filename} did not detect any text. Deleting")
                    with open("delete.txt", "a") as file:
                        file.write(f"{filename}\n")
                        file.close()
                    return {}
                pdf_file.close()

                # Extract metadata
                metadata = self.get_file_metadata(filename)
                metadata_size = str(metadata['size'])  # Has to be str or int for some reason, was not working
                metadata_created_at = metadata['created_at']

                text = self._clean_text(text)

                # Cut down text from PDF into reasonable chunks
                chunks = []
                chunked_text = []
                start_idx = 0
                count = 0
                while start_idx < len(text):
                    count += 1
                    end_idx = start_idx + self.max_chunk_size
                    if end_idx >= len(text):
                        tmp = text[start_idx:]
                    else:
                        tmp = text[start_idx:end_idx]

                    chunks.append({
                        'id': str(self.doc_count),
                        'metadata': {
                            'filename': filename,
                            'size': metadata_size,
                            'created_at': metadata_created_at,
                            'chunk': count
                        }
                    })
                    chunked_text.append(tmp)
                    self.doc_count = self.doc_count + 1
                    start_idx = end_idx - self.chunk_overlap

                vectorized_chunks = encoder.encode(chunked_text, show_progress_bar=False)
                for i, chunk in enumerate(chunks):
                    chunk['values'] = vectorized_chunks[i].tolist()
                    chunk['metadata']['text'] = chunked_text[i]

                return chunks
        except Exception as e:
            self.logger.error(f"An error occurred: {e}")
            return {}

    def updateWorkers(self, newValue: int) -> None:
        """Method to safely update the number of workers

        :param newValue: New num of workers
        :return:
        """
        self.max_workers_num = newValue
        if self.device == 'cuda':

            # If the array already has more values than needed, cut them. Else add them until satisfied.
            # Saves time as each SentenceTransformer can take a moment, Rarely will affect a lot but important to consider.
            if len(self.embeddings) > newValue:
                self.embeddings = self.embeddings[:newValue]
            else:
                while newValue > len(self.embeddings):
                    self.embeddings.append(SentenceTransformer(model_name_or_path=self.model_name, device=self.device))
