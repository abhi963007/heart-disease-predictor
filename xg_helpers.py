import urllib.request
import zipfile
import pandas as pd

from sklearn.metrics import accuracy_score
from typing import Any, Dict, Union
import xgboost as xgb

def extract_zip(src, dst, member_name):
    """
    Extract a member file from a zip file and read it into a pandas DataFrame.

    Parameters:
        src (str): URL of the zip file to be downloaded and extracted.
        dst (str): Local file path where the zip file will be written.
        member_name (str): Name of the member file inside the zip file to be read into a DataFrame.

    Returns:
        pandas.DataFrame: DataFrame containing the contents of the member file.
    """
    # Download the zip file from the given URL
    url = src
    fname = dst
    fin = urllib.request.urlopen(url)
    data = fin.read()
    
    # Write the downloaded data to the specified local file
    with open(dst, mode='wb') as fout:
        fout.write(data)
    
    # Extract the specified member file from the zip and read it into a DataFrame
    with zipfile.ZipFile(dst) as z:
        kag = pd.read_csv(z.open(member_name))
        kag_questions = kag.iloc[0]  # Extracting the first row (might be headers or metadata)
        raw = kag.iloc[1:]  # Extracting all rows after the first one
    
    return raw
