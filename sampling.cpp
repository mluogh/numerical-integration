#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <stdlib.h>
#include <random>
#include <thread>
#include <future>

namespace py = pybind11;

typedef unsigned int uint;

double _sample_hypercube(py::function fn, std::vector<double> lows, std::vector<double> highs, unsigned int n) {
    std::random_device rd;
    unsigned int seed = getenv("UNITTESTING") == NULL ? rd() : 0;
    std::mt19937 gen(seed);
    std::vector<std::uniform_real_distribution<double>> uniform_dists;
    double count = 0;
    unsigned int dim = highs.size();

    for (unsigned int i = 0; i < dim; i++){
        uniform_dists.push_back(std::uniform_real_distribution<double>(lows[i], highs[i]));
    }

    uint CHUNK=1000;

    std::vector<std::vector<double>> xs(CHUNK, std::vector<double>(dim));

    for (unsigned int c = 0; c < (uint)(n / CHUNK); c++) {
        for (unsigned int j = 0; j < CHUNK; j++) {
            for (unsigned int i = 0; i < dim; i++) {
                xs[j][i] = uniform_dists[i](gen);
            }
            count += fn(xs[j]).cast<double>();
        }
    }

    return count;
}

std::vector<py::array> _metropolis_hastings(py::array x, py::function proposal_fn, py::function acceptance_fn, unsigned int n, unsigned int burn_in, unsigned int skip) {
    std::random_device rd;
    unsigned int seed = getenv("UNITTESTING") == NULL ? rd() : 0;
    std::mt19937 gen(seed);
    std::uniform_real_distribution<> dis(0.0, 1.0);

    std::vector<py::array> vals;

    for (uint i = 0; i < burn_in + (n * skip); i++) {
        py::array x_p = proposal_fn(x).cast<py::array>();

        double a = std::min(1.0, acceptance_fn(x, x_p).cast<double>()); 

        if (dis(gen) < a) {
            x = x_p;
        }

        if (i >= burn_in && (i - burn_in) % skip == 0){
            vals.push_back(x);
        }
    }
    return vals; 
}

PYBIND11_MODULE(sampling, m) {
    m.doc() = "C++ bindings for numerical integration library"; // optional module docstring

    m.def("_sample_hypercube", &_sample_hypercube, "Samples uniformly from some hypercube");
    m.def("_metropolis_hastings", &_metropolis_hastings, "Generate samples from an arbitrary pdf");
}
