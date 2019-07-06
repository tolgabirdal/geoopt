#pragma once
#include <torch/extension.h>

template <class c_t>
torch::Tensor lambda_x(const torch::Tensor x, const c_t c, const int dim, const bool keepdim);
