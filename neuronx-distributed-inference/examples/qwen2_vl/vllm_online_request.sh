
export MODEL_PATH="/home/ubuntu/qwenvl/qwen-vl-main/neuronx-distributed-inference/examples/qwen2_vl/models/Qwen/Qwen2-VL-7B-Instruct"

#base64 -i image_1024_768.jpg | tr -d '\n' > image_base64.txt
base64 -i dog.jpg | tr -d '\n' > image_base64.txt

BASE64_IMAGE=$(cat image_base64.txt)

cat > request.json << EOF
{
  "model": "${MODEL_PATH}",
  "max_tokens": 10,
  "stream":false,
  "messages": [
    {
      "role": "user", 
      "content": [
        {"type": "text", "text": "Tell me a story about this picture"}, 
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,${BASE64_IMAGE}"}},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,${BASE64_IMAGE}"}}
      ]
    }
  ]
}
EOF

curl -w 'Total: %{time_total}s\n' http://0.0.0.0:8080/v1/chat/completions   \
-H "Content-Type: application/json"   \
-d @request.json

