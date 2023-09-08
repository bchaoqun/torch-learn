#include <iostream>

#include "Preliminaries.h"

using namespace std;

int main() {
    //    Preliminaries::Preliminaries::ndarray();
    torch::Tensor output;
    cout << "cuda is_available: " << torch::cuda::is_available() << endl;
    torch::DeviceType device = at::kCPU;
    if (torch::cuda::is_available()) device = at::kCUDA;
    output = torch::randn({3, 3}).to(device);
    cout << output << endl;
    return 0;
}
