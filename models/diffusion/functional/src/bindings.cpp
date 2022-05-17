#include <pybind11/pybind11.h>

#include "ball_query.hpp"

PYBIND11_MODULE(ball_query, m) {
  m.def("ball_query", &ball_query_forward, "Ball Query (CUDA)");
}
