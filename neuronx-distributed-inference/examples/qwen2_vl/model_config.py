num_img_per_prompt=2 #1 2 4 8
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "/home/ubuntu/qwenvl/qwen-vl-main/neuronx-distributed-inference/examples/qwen2_vl/dog.jpg"}
            for _ in range(num_img_per_prompt)
        ] + [
            {"type": "text", "text": "Describe what you see in the images"}
        ],
    }
]
