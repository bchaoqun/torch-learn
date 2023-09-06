#include <iostream>

#include <torch/torch.h>
#include <pybind11/embed.h>

int main() {
    torch::Tensor tensor = torch::rand({2, 3});
    std::cout << tensor << std::endl;

    pybind11::scoped_interpreter guard{};
    pybind11::module matplotlib = pybind11::module::import("matplotlib");
    pybind11::print("matplotlib version: ", matplotlib.attr("__version__"));
    return 0;
}
