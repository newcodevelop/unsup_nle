import os
import zipfile
import urllib.request
from tqdm import tqdm

val_url = "http://images.cocodataset.org/zips/val2014.zip"
val_dir = "./val2014"
os.makedirs(val_dir, exist_ok=True)

def download_with_progress(url, filename):
    response = urllib.request.urlopen(url)
    total = int(response.getheader('Content-Length').strip())

    with open(filename, 'wb') as f, tqdm(
        desc=f"Downloading {os.path.basename(filename)}",
        total=total,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        while True:
            chunk = response.read(8192)
            if not chunk:
                break
            f.write(chunk)
            bar.update(len(chunk))

def download_and_extract_val(url):
    filename = os.path.basename(url)

    if not os.path.exists(filename):
        download_with_progress(url, filename)
    else:
        print(f"{filename} already exists.")

    print(f"Extracting {filename}...")
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall(".")  
    print(f"Extracted to ./val2014")

download_and_extract_val(val_url)

