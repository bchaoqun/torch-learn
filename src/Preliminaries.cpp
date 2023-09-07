//
// Created by 毕超群 on 2023/9/6.
//

#include "Preliminaries.h"

namespace Preliminaries {
    void Preliminaries::ndarray() {
        // 张量表示一个数值组成的数组，这个数组可能有多个维度 张量 向量 多维数组
        torch::Tensor x = torch::arange(12);
        std::cout << "arange: " << '\n'
                  << x << '\n';
        std::cout << "----------------------------------------------------------------" << '\n';
        // 通过sizes属性来访问张量的形状和元素总数
        std::cout << "x sizes: " << '\n'
                  << x.sizes() << '\n';// python中为shape
        std::cout << "x numel: " << '\n'
                  << x.numel() << '\n';// number of elements
        std::cout << "----------------------------------------------------------------" << '\n';
        // 改变一个张量的形状而不改变元素值，使用reshape
        x = x.reshape({3, 4});
        std::cout << "x reshape: " << '\n'
                  << x << '\n';
        std::cout << "----------------------------------------------------------------" << '\n';
        // 便捷创建全0和全1或其他常亮或从特定分布中随机采样的数字张量
        x = torch::zeros({2, 3, 4});
        std::cout << "x zeros: " << '\n'
                  << x << '\n';
        x = torch::ones({2, 3, 4});
        std::cout << "x ones: " << '\n'
                  << x << '\n';
        x = torch::rand({2, 3, 4});
        std::cout << "x rand: " << '\n'
                  << x << '\n';
        std::cout << "----------------------------------------------------------------" << '\n';
        // 从容器中创建张量
        std::vector<int64_t> xs = {1, 2, 3, 4};
        x = torch::tensor(xs);
        std::cout << "x tensor: " << '\n'
                  << x << '\n';
        float arr[5] = {1, 2, 3, 4, 5};
        x = torch::from_blob(arr, {1, 5}, torch::kFloat32);
        std::cout << "x tensor: " << '\n'
                  << x << '\n';
        x = torch::tensor({{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}});
        std::cout << "x tensor: " << '\n'
                  << x << '\n';
        x = torch::tensor({{1}, {5}, {9}, {13}});
        std::cout << "x tensor: " << '\n'
                  << x << '\n';
        std::cout << "----------------------------------------------------------------" << '\n';
        // 张量的标准算数运算都可以被升级为按元素运算
        torch::Tensor tensor_x = torch::tensor({1, 2, 4, 8});
        torch::Tensor tensor_y = torch::tensor({2, 2, 2, 2});
        std::cout << "tensor_x + tensor_x: " << '\n'
                  << tensor_x + tensor_y << '\n';
        std::cout << "tensor_x - tensor_x: " << '\n'
                  << tensor_x - tensor_y << '\n';
        std::cout << "tensor_x * tensor_x: " << '\n'
                  << tensor_x * tensor_y << '\n';
        std::cout << "tensor_x / tensor_x: " << '\n'
                  << tensor_x / tensor_y << '\n';
        std::cout << "tensor_x ** tensor_x: " << '\n'
                  << pow(tensor_x, tensor_y) << '\n';
        std::cout << "----------------------------------------------------------------" << '\n';
        // 张量连接
        torch::Tensor tensor_x1 = torch::arange(12, torch::kFloat32).reshape({3, 4});
        torch::Tensor tensor_x2 = torch::tensor({{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}});
        torch::Tensor tensor_x1_x2_dim0 = torch::cat({tensor_x1, tensor_x2}, 0);
        torch::Tensor tensor_x1_x2_dim1 = torch::cat({tensor_x1, tensor_x2}, 1);
        std::cout << "tensor_x1_x2_dim0: " << '\n'
                  << tensor_x1_x2_dim0 << '\n';
        std::cout << "tensor_x1_x2_dim1: " << '\n'
                  << tensor_x1_x2_dim1 << '\n';
        std::cout << "----------------------------------------------------------------" << '\n';
        // 通过逻辑运算符构建二元张量
        torch::Tensor logi = tensor_x1 == tensor_x2;
        std::cout << "logi: " << '\n'
                  << logi << '\n';
        std::cout << "----------------------------------------------------------------" << '\n';
        // 对张量中所有元素求和会产生只有一个元素的张量,没有行和列
        torch::Tensor sum = tensor_x1.sum();
        std::cout << "sum: " << '\n'
                  << sum << '\n';
        std::cout << "shape: " << '\n'
                  << sum.sizes() << '\n';
        std::cout << "----------------------------------------------------------------" << '\n';
        // 两个张量之间的运算会调用广播机制 broadcasting mechanism,广播机制会自动将维度较小的张量复制到维度较大的张量中
        torch::Tensor a = torch::arange(3).reshape({3, 1});
        torch::Tensor b = torch::arange(2).reshape({1, 2});
        std::cout << a << '\n';
        std::cout << b << '\n';
        std::cout << "a + b: " << '\n'
                  << a + b << '\n';
        std::cout << "----------------------------------------------------------------" << '\n';

    }
}// namespace Preliminaries