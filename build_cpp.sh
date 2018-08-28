c++ -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` sampling.cpp -o sampling`python3-config --extension-suffix`

