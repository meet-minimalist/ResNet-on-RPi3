# ResNet-on-RPi3
Running ResNet50 model on RaspberryPi 3B+

# ONNX Runtime binaries for RaspberryPi 3B+ (Bookworm) : 
[Download](https://github.com/ava-orange-education/Ultimate-ONNX-for-Optimizing-Deep-Learning-Models/releases/tag/v1.22.0)

# Commands:
```
# Download ONNX Runtime Prebuilt binaries
wget https://github.com/ava-orange-education/Ultimate-ONNX-for-Optimizing-Deep-Learning-Models/releases/download/v1.22.0/onnxruntime-linux-aarch64-1.22.0.zip
tar -xzvf onnxruntime-linux-x64-1.22.0.tgz

mkdir deps/
mkdir data/

# Download image processing helper libraries
wget https://raw.githubusercontent.com/nothings/stb/master/stb_image.h -O deps/stb_image.h
wget https://raw.githubusercontent.com/nothings/stb/master/stb_image_resize2.h -O deps/stb_image_resize2.h

# Download ResNet50 model from Onnx model zoo
wget https://github.com/onnx/models/raw/refs/heads/main/validated/vision/classification/resnet/model/resnet50-v1-12.onnx -O ./data/resnet50-v1-12.onnx
wget https://github.com/onnx/models/raw/refs/heads/main/validated/vision/classification/resnet/model/resnet50-v1-12-qdq.onnx -O ./data/resnet50-v1-12-qdq.onnx 
wget https://github.com/onnx/models/raw/refs/heads/main/validated/vision/classification/resnet/model/resnet50-v1-12-int8.onnx -O ./data/resnet50-v1-12-int8.onnx

# Download a sample image of cat
wget https://huggingface.co/spaces/ClassCat/ViT-ImageNet-Classification/resolve/main/samples/cat.jpg -O ./data/cat.jpg

# Build app
g++ -std=c++17 -I./onnxruntime-linux-x64-1.22.0/include -I./deps/ -L./onnxruntime-linux-x64-1.22.0/lib -o resnet_inference resnet_inference.cpp -lonnxruntime

# Add ONNX Runtime library path in LD_LIBRARY_PATH as we have dynamically linked this library during compilation and Run the inference
!LD_LIBRARY_PATH=./onnxruntime-linux-x64-1.22.0/lib:$LD_LIBRARY_PATH 

# Run the application
./resnet_inference ./data/resnet50-v1-12.onnx ./data/cat.jpg
./resnet_inference ./data/resnet50-v1-12-int8.onnx ./data/cat.jpg
./resnet_inference ./data/resnet50-v1-12-qdq.onnx ./data/cat.jpg
```


# Comparision
| Sr. No. | Model Name | ORT session.run time |
|--|--|--|
| 1 | [ResNet50 - FP32](https://github.com/onnx/models/raw/refs/heads/main/validated/vision/classification/resnet/model/resnet50-v1-12.onnx) | 2.57080 sec |
| 2 | [ResNet50 - INT8](https://github.com/onnx/models/raw/refs/heads/main/validated/vision/classification/resnet/model/resnet50-v1-12-int8.onnx) | 1.16976 sec |
| 3 | [ResNet50 - QDQ](https://github.com/onnx/models/raw/refs/heads/main/validated/vision/classification/resnet/model/resnet50-v1-12-qdq.onnx) | 1.21838 sec |
Note: The numbers are obtained from Raspberry Pi 3B+ (Bookworm).


# References:
- Ultimate ONNX for Optimizing Deep Learning Models : [Github](https://github.com/ava-orange-education/Ultimate-ONNX-for-Optimizing-Deep-Learning-Models)