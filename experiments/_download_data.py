import os
import subprocess
import zipfile
from decoding import config
from datalad.api import install, get


def download_and_unzip(url, output_filename, extract_path):
    # Ensure extract path exists
    os.makedirs(extract_path, exist_ok=True)

    # Download file with wget
    subprocess.run(["wget", "-O", output_filename, url], check=True)

    # Unzip
    with zipfile.ZipFile(output_filename, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    # Delete the zip file
    os.remove(output_filename)

    print(f"Downloaded and extracted to {extract_path}, removed {output_filename}")

def download_and_get_derivative(url_datalad, local_path_to_derivative):
    dataset = install(
        source=url_datalad,
        path=os.path.dirname(local_path_to_derivative),
    )
    get(local_path_to_derivative)  # get all content


if __name__ == "__main__":
    download_and_unzip(
        url='https://utexas.box.com/shared/static/7ab8qm5e3i0vfsku0ee4dc6hzgeg7nyh.zip',
        output_filename="data_lm_tmp.zip",
        extract_path=config.DATA_LM_DIR,
    )

    download_and_unzip(
        url='https://utexas.box.com/shared/static/3go1g4gcdar2cntjit2knz5jwr3mvxwe.zip',
        output_filename="data_train_tmp.zip",
        extract_path=config.DATA_TRAIN_DIR,
    )

    download_and_unzip(
        url='https://utexas.box.com/shared/static/ae5u0t3sh4f46nvmrd3skniq0kk2t5uh.zip',
        output_filename="data_test_tmp.zip",
        extract_path=config.DATA_TEST_DIR,
    )

    download_and_get_derivative(
        url_datalad='https://github.com/OpenNeuroDatasets/ds004510.git',
        local_path_to_derivative=config.DATA_PATH_TO_DERIVATIVE_DS004510,
    )

    download_and_get_derivative(
        url_datalad='https://github.com/OpenNeuroDatasets/ds003020.git',
        local_path_to_derivative=config.DATA_PATH_TO_DERIVATIVE_DS003020,
    )

    




    


