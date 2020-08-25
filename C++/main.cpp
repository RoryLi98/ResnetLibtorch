#include <torch/script.h>
#include <torch/torch.h> 
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

/* main */
int main() {

    // Deserialize the ScriptModule from a file using torch::jit::load()
    torch::jit::script::Module module;
    try {
        module = torch::jit::load("/home/link/NetworkModel1/model1111.pt");
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        return -1;
    }
    std::cout << "load model ok\n";

    // load image with opencv and transform
    cv::Mat image;
    image = cv::imread("/home/link/NN_Release/C++/3.png");
    cv::cvtColor(image, image, CV_BGR2GRAY);
    cv::resize(image, image, cv::Size(224, 224));

    torch::Tensor img_tensor = torch::from_blob(image.data, { 224,224, 1}, torch::kByte);
    img_tensor = img_tensor.permute({2,0,1});
    img_tensor = img_tensor.toType(torch::kFloat);
    img_tensor = img_tensor.div(255);
    img_tensor = img_tensor.unsqueeze(0);

    torch::Tensor output = module.forward({img_tensor}).toTensor();

//    auto max_result = output.max(1, true);
//    auto max_index = std::get<1>(max_result).item<float>();
//    std::cout << output << std::endl;
//    std::cout << max_index << std::endl;

    // print predicted top-5 labels
    std::tuple<torch::Tensor,torch::Tensor> result = output.sort(-1, true);
    torch::Tensor top_scores = std::get<0>(result)[0];
    torch::Tensor top_idxs = std::get<1>(result)[0].toType(torch::kInt32);
  
    auto top_scores_a = top_scores.accessor<float,1>();
    auto top_idxs_a = top_idxs.accessor<int,1>();

     for (int i = 0; i < 5; ++i)
     {
         int idx = top_idxs_a[i];
         std::cout << "top-" << i+1 << " label: ";
         std::cout << idx << ", score: " << top_scores_a[i] << std::endl;
     }

     return 1;

}

