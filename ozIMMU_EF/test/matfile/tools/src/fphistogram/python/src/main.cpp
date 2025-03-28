#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <fphistogram/fphistogram.hpp>

template <class T>
void print_histogram(
	pybind11::array_t<T, pybind11::array::c_style | pybind11::array::forcecast> array,
	const unsigned num_all_stars = 100,
	const unsigned base = 2
	) {
	const auto info = array.request();
	auto ptr = static_cast<T*>(info.ptr);

	// get size
	unsigned n = 1;
	for (const auto r : info.shape) {
		n *= r;
	}

	// call print func
	if (base == 2) {
		mtk::fphistogram::print_histogram<T, mtk::fphistogram::mode_log2 >(ptr, n, num_all_stars);
	} else if (base == 10) {
		mtk::fphistogram::print_histogram<T, mtk::fphistogram::mode_log10>(ptr, n, num_all_stars);
	} else {
		std::fprintf(stderr, "Not supported : base = %u\n", base);
	}
}

template <class T>
void print_histogram_pm(
	pybind11::array_t<T, pybind11::array::c_style | pybind11::array::forcecast> array,
	const unsigned num_all_stars = 100,
	const unsigned base = 2
	) {
	const auto info = array.request();
	auto ptr = static_cast<T*>(info.ptr);

	// get size
	unsigned n = 1;
	for (const auto r : info.shape) {
		n *= r;
	}

	// call print func
	if (base == 2) {
		mtk::fphistogram::print_histogram_pm<T, mtk::fphistogram::mode_log2 >(ptr, n, num_all_stars);
	} else if (base == 10) {
		mtk::fphistogram::print_histogram_pm<T, mtk::fphistogram::mode_log10>(ptr, n, num_all_stars);
	} else {
		std::fprintf(stderr, "Not supported : base = %u\n", base);
	}
}

PYBIND11_MODULE(fphistogram, m) {
    m.doc() = "fp-histogram for Python";

    m.def("print_histogram"    , &print_histogram<double>   , "print_histogram"   , pybind11::arg("array"), pybind11::arg("num_all_stars") = 100, pybind11::arg("base") = 2);
    m.def("print_histogram"    , &print_histogram<float>    , "print_histogram"   , pybind11::arg("array"), pybind11::arg("num_all_stars") = 100, pybind11::arg("base") = 2);
    m.def("print_histogram_pm" , &print_histogram_pm<double>, "print_histogram_pm", pybind11::arg("array"), pybind11::arg("num_all_stars") = 100, pybind11::arg("base") = 2);
    m.def("print_histogram_pm" , &print_histogram_pm<float> , "print_histogram_pm", pybind11::arg("array"), pybind11::arg("num_all_stars") = 100, pybind11::arg("base") = 2);
}
