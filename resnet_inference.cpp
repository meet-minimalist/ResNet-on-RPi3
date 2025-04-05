#include <iostream>
#include <vector>
#include <chrono>
#include <onnxruntime_cxx_api.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_RESIZE2_IMPLEMENTATION
#include <stb_image.h>
#include <stb_image_resize2.h>


// ImageNet mean and std values
const float mean[] = {0.485f, 0.456f, 0.406f};
const float std_dev[] = {0.229f, 0.224f, 0.225f};

void preprocessImage(const std::string& imagePath, std::vector<float>& inputTensorValues) {
    int width, height, channels;
    unsigned char* img = stbi_load(imagePath.c_str(), &width, &height, &channels, 3); // Load as RGB

    if (!img) {
        throw std::runtime_error("Failed to load image: " + imagePath);
    }

    int target_width = 224, target_height = 224;
    std::vector<unsigned char> resized_img(target_width * target_height * 3);

    // Resize image to 224x224
    stbir_resize_uint8_linear(img, width, height, 0,
                       resized_img.data(), target_width, target_height, 0, STBIR_RGB);

    // Convert to normalized float format (CHW layout)
    for (int c = 0; c < 3; c++) {  // Loop over channels
        for (int h = 0; h < target_height; h++) {
            for (int w = 0; w < target_width; w++) {
                int idx = (h * target_width + w) * 3 + c;  // HWC index
                inputTensorValues[c * target_width * target_height + h * target_width + w] =
                    ((float)resized_img[idx] / 255.0f - mean[c]) / std_dev[c];  // Normalize & Standardize
            }
        }
    }

    stbi_image_free(img);
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model_path.onnx> <image_path.jpg>" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];
    std::string img_path = argv[2];
    
    int model_height = 224;
    int model_width = 224;
    int num_channels = 3;

    try {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNXRuntimeTest");
        Ort::SessionOptions session_options;

        // Load model
        Ort::Session session(env, model_path.c_str(), session_options);
        Ort::AllocatorWithDefaultOptions allocator;

        // Get input details
        auto input_name = session.GetInputNameAllocated(0, allocator);
        std::vector<int64_t> input_shape = {1, num_channels, model_height, model_width}; // Batch size 1
        
        // Preprocess image
        std::vector<float> input_tensor_values(num_channels * model_height * model_width, 0.0f);
        preprocessImage(img_path, input_tensor_values);
        
        // Create input tensor
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(),
                            input_tensor_values.size(), input_shape.data(), input_shape.size());
        
        // Run inference
        std::vector<const char*> input_names = {input_name.get()};
        auto output_name = session.GetOutputNameAllocated(0, allocator);
        std::vector<const char*> output_names = {output_name.get()};
        
        int num_iter = 10;
        std::vector<Ort::Value> output_tensors;
        
        std::cout << "Execution started.\n";
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_iter; ++i) {
            output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names.data(),
                                                                    &input_tensor, 1, output_names.data(), 1);
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        std::cout << "ORT Inference Time: " << duration.count() / num_iter << " seconds\n";


        // Print top-1 predicted class index
        float* output_data = output_tensors[0].GetTensorMutableData<float>();

        int predicted_class = std::max_element(output_data, output_data + 1000) - output_data;
        std::cout << "Predicted Class: " << predicted_class << std::endl;

    } catch (const Ort::Exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
