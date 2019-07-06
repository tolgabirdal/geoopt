#include "poincare.h"

template <class c_t>
torch::Tensor lambda_x(const torch::Tensor x, const c_t c, const int dim, const bool keepdim) {
    auto denom = (1 - c * torch::sum(torch::pow(x, 2), dim, keepdim));
    return 2 / torch::clamp_min(denom, 1e-5);
};


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("lambda_x", &lambda_x<torch::Tensor>, "lambda_x forward",
        py::arg("x"), py::arg("c"), py::arg("dim")=-1, py::arg("keepdim")=false
    );
    m.def("lambda_x", &lambda_x<float>, "lambda_x forward",
        py::arg("x"), py::arg("c")=1., py::arg("dim")=-1, py::arg("keepdim")=false
    );
}
