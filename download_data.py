import requests
import config
import zipfile

def download_file_from_google_drive(id, destination):
    """Copied from some Github repo"""

    URL = "https://docs.google.com/uc?export=download"

    print("Downloading data...")

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)


def download():
    download_file_from_google_drive(config.file_id, config.download_dest)
    with zipfile.ZipFile(config.download_dest, "r") as zip_ref:
        zip_ref.extractall(config.DATA_PATH)