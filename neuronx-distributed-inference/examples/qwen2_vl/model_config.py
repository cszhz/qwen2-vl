num_img_per_prompt=128 #1 2 4 8
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "/home/ubuntu/qwen2-vl/neuronx-distributed-inference/examples/qwen2_vl/dog_640_480.jpg"}
            for _ in range(num_img_per_prompt)
        ] + [
            {"type": "text", "text": "Describe what you see in the images"}
        ],
    }
]
