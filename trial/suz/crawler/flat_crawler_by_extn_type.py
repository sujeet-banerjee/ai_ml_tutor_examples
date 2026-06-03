import os
import requests
import xml.etree.ElementTree as ET
from urllib.parse import quote


def main():
    base_url = "https://storage.googleapis.com/vetting-engine-apt-static"

    # 1. Get the desired file extension from the user
    target_ext = (input(
        "Enter the file extension to download (e.g., .pdf, .png): ").
                  strip().lower())

    # Ensure it starts with a dot
    if not target_ext.startswith('.'):
        target_ext = '.' + target_ext

    print(f"\nFetching bucket index from {base_url}...")

    # 2. Fetch the XML index
    try:
        response = requests.get(base_url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching the index: {e}")
        return

    # 3. Parse the XML securely
    try:
        root = ET.fromstring(response.content)
    except ET.ParseError as e:
        print(f"Error parsing XML: {e}")
        return

    # S3/GCS XML uses a specific namespace. We must define it to find the tags.
    namespace = {'s3': 'http://doc.s3.amazonaws.com/2006-03-01'}

    # 4. Extract and filter the keys
    filtered_keys = []

    # Find all <Contents> tags using the namespace
    for contents in root.findall('s3:Contents', namespace):
        key_element = contents.find('s3:Key', namespace)

        if key_element is not None and key_element.text:
            key = key_element.text
            # Check if it matches our target extension
            if key.lower().endswith(target_ext):
                filtered_keys.append(key)

    if not filtered_keys:
        print(f"No files found with the extension '{target_ext}'.")
        return

    print(f"Found {len(filtered_keys)} file(s) matching '{target_ext}'.")

    # 5. Create a local directory to save the files
    download_dir = "C:\Sujeet\BOOKS_ASSORTED\crawled\downloaded_files"
    os.makedirs(download_dir, exist_ok=True)
    print(f"Saving files to local directory: ./{download_dir}/\n")

    # 6. Download the files
    for key in filtered_keys:
        # urllib.parse.quote correctly encodes spaces (%20) and special characters
        encoded_key = quote(key)
        file_url = f"{base_url}/{encoded_key}"

        # Create a safe local filename (in case keys contain nested folder slashes like 'folder/file.pdf')
        safe_filename = key.replace('/', '_')
        filepath = os.path.join(download_dir, safe_filename)

        print(f"Downloading: {key} ...")

        try:
            # stream=True ensures we don't load massive files into RAM all at once
            file_response = requests.get(file_url, stream=True)
            file_response.raise_for_status()

            with open(filepath, 'wb') as f:
                for chunk in file_response.iter_content(chunk_size=8192):
                    f.write(chunk)

        except requests.exceptions.RequestException as e:
            print(f" -> Failed to download {key}: {e}")

    print("\n✅ All downloads complete.")


if __name__ == "__main__":
    main()