import logging
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import pyodbc

write_path = str(Path.home()) + '\\RADIX ENGENHARIA E DESENVOLVIMENTO DE SOFTWARE S A (ISV)\\EPASA-P&D - General\\dataset\\01.real\\epm'

server = '20.186.181.130' 
database = 'EPM_Database1' 
username = 'epasa' 
password = 'ep@s@#062020' 

## Def Logger
logger = logging.getLogger("epasa_epm_logger")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('application.log')
pattern = '%(asctime)s : %(levelname)s : %(name)s : %(message)s'
formatter = logging.Formatter(pattern)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def _get_all_files_direcotry():
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(write_path):
        for file in f:
            if '.xz' in file:
                files.append(os.path.join(r, file))
    return files

def check_files_consistency():
    files = _get_all_files_direcotry()
    for xz_file in files:
        try:
            output = pd.read_csv(xz_file, compression='xz')
            if output.size == 0 or  len(output.columns) != 3:
                os.remove(xz_file)
                logger.info("Removed file: {}".format(xz_file))
        except Exception:
            os.remove(xz_file)
            logger.info("Removed file: {}".format(xz_file))
        print("Checking file: {}".format(xz_file))
        
def get_tag_list():
    cnxn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password)
    df_tags = pd.read_sql_query("select * from Epm_Tags", cnxn)
    tags_list = df_tags['Name'].unique()
    cnxn.close()
    tags_list = np.sort(tags_list)
    return tags_list

def worker(tag):
    if "_" in tag:
        tag_split = tag.strip().split('_')
        if (not os.path.exists('{}/{}/'.format(write_path,tag_split[0]))):
            os.mkdir('{}/{}/'.format(write_path,tag_split[0]))
        if (not os.path.exists('{}/{}/{}.tar.xz'.format(write_path,tag_split[0], tag))):
            print("Download Tag: {}".format(tag))
            try:
                logger.info("Download Tag: {}".format(tag))
                cnxn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password)
                query = "SELECT CAST(A.Timestamp as datetime) as DataHora, CAST(A.Value as NUMERIC(10,2)) AS '{}' FROM dbo.EpmQueryRawFunction(-3,'01/01/2018 00:00:00','05/30/2020 00:00:00',0,0,'{}') AS A".format(tag, tag)
                unidade_geradora = pd.read_sql_query(query, cnxn)
                unidade_geradora.to_csv('{}/{}/{}.tar.xz'.format(write_path,tag_split[0], tag), compression='xz')
                cnxn.close()
            except Exception:
                logger.error("Tag With Problem: {}".format(tag))
                return False
        return True
    else:
        logger.error("Skipped Tag: {}".format(tag))
        return False

if __name__ ==  '__main__': 
    num_processors = 8
    check_files_consistency()
    p=Pool(processes = num_processors)
    tags_list = get_tag_list()
    p.map(worker,tags_list)
