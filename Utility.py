import logging
import os


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
