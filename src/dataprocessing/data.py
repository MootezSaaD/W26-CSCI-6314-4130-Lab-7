'''
Author: Indranil Palit
Created: 27-12-20223
Description: This script is used to process the repositories and extract the methods from the repositories.
'''
import subprocess
import os
import threading
import time
import json
import logging
import sys
import concurrent.futures


def generate_repository_details(input_file, count = -1):
    '''Generate repository details from the input file'''

    with open(input_file,'r', errors='ignore') as f: #pylint: disable=unspecified-encoding
        data = json.load(f)
        print(f"Total repositories: {count}")
        if count != -1:
            _counter = 0
        for i, item in enumerate(data.get('items',[])):
            yield item
            if count != -1:
                _counter += 1
                if _counter == count:
                    break


lock = threading.Lock()

def process_repositories(item):
    '''Process the repository'''
    GITHUB_BASE_URL = "https://github.com/"
    # GITHUB_BASE_URL= "git@github.com:"
    name = item.get('name')
    repository_name = name.split('/')[-1]
    default_branch = item.get('defaultBranch')

    jar_path = os.path.join(os.getcwd(),"refminer-extractmethod","target","extract-method-extractor-1.0-jar-with-dependencies.jar")
    output_path = os.path.join(os.getcwd(),"data","output",repository_name+".jsonl")
    repo_url = f"{GITHUB_BASE_URL}{name}.git"

    os.makedirs(os.path.join(os.getcwd(),"data","output","logs"),exist_ok=True)
    log_file = os.path.join(os.getcwd(),"data","output","logs", "log.txt")
    logging.basicConfig(filename=log_file, level=logging.INFO)
    try:
        result = subprocess.run(['java','-jar',jar_path,repo_url,output_path,default_branch],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        with lock:
            if result.returncode == 0:
                logging.info("Successfully processed %s", name)
                print("Successfully processed %s", name)
            else:
                logging.error(result.stderr.decode())
    except subprocess.CalledProcessError as e:
        with lock:
            logging.error(e.stderr.decode())
        return {"result":e.returncode, "name":name}

    return {"result":result.returncode, "name":name}

if __name__=="__main__":

    print("Start processing")
    ti = time.time()
    input_file_path = sys.argv[1]
    json_file_path = os.path.join(os.getcwd(), input_file_path)

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        repository_generator = generate_repository_details(json_file_path, count=5)
        output = executor.map(process_repositories, repository_generator)

    output_file_path = os.path.join(os.getcwd(), "data", "output", "output.json")

    with open(output_file_path, 'w') as fp: #pylint: disable=unspecified-encoding
        json.dump(list(output), fp)

    print("Time taken: ", time.time()-ti)

    print("Output saved as:", output_file_path)
