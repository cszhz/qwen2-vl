from huggingface_hub import snapshot_download
model_id='Qwen/Qwen2-VL-7B-Instruct'

snapshot_download(repo_id=model_id,local_dir="./models/"+model_id)
