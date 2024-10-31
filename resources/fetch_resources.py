# Python script that fetches a canonical set of T-Res resources
# from an Azure storage container.

import os
import sys
import zipfile
from pathlib import Path

# Check azcopy is available.
azcopy_check = os.system("azcopy --version")
if azcopy_check != 0:
    sys.exit("Please install AzCopy.")
print("This script will download and unzip an archive containing T-Res resources.\n")
print("The archive is large (>8GB) and therefore this process may take some time.\n\n")
sas_token = input("Please enter a SAS token to access the T-Res resources container:\n")

# Download the zip archive.
local_path = Path(os.path.dirname(os.path.abspath(__file__)))
remote = f'"https://tres.blob.core.windows.net/resources/resources.zip?{sas_token}"'
azcopy_command = f"azcopy copy {remote} {local_path}"
print(azcopy_command)
os.system(azcopy_command)

resources_zip = Path(local_path, "resources.zip")
if not resources_zip.is_file():
    sys.exit("Failed to download resources archive.")

# Unzip the archive.
print("Unzipping (please be patient...)")
with zipfile.ZipFile(resources_zip, 'r') as zip_ref:
    zip_ref.extractall(local_path)

# Clean up.
Path.unlink(resources_zip)
