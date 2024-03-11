try:
    import click
    from google.cloud import storage
except Exception:
    print("Please ensure that google-cloud-storage and/or click are installed, else pip install.")
    
def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket. https://cloud.google.com/storage/docs/ """
    storage_client = storage.Client.create_anonymous_client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print('File {} uploaded to {}.'.format(
        source_file_name,
        destination_blob_name))
    
def download_public_file(bucket_name, source_blob_name, destination_file_name):
    """Downloads a public blob from the bucket."""
    # bucket_name = "your-bucket-name"
    # source_blob_name = "storage-object-name"
    # destination_file_name = "local/path/to/file"

    storage_client = storage.Client.create_anonymous_client()

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    print(
        "Downloaded public blob {} from bucket {} to {}.".format(
            source_blob_name, bucket.name, destination_file_name
        )
    )

@click.command()
@click.option("-o", "--option", required=True, type=str, help="download/upload option only")
@click.option("--bucket_path", required=True, type=str, help="Example: ddi_kt_2024_multimodal_research/no_padding_customdataset/biobert_cased_mean/train.pt")
@click.option("--local_path", required=True, type=str, help="Example: /kaggle/working/DDI-KT-2024/train.pt")
def process(option, bucket_path, local_path):
    bucket_name = bucket_path.split("/")[0]
    source_blob_name = "/".join(bucket_path.split("/")[1:])
    if option == "download":
        download_public_file(bucket_name, source_blob_name, local_path)
    if option == "upload":
        upload_blob(bucket_name, source_file_name, local_path)

if __name__=="__main__":
    process()