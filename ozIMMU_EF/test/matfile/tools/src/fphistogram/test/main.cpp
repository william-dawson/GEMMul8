#include <iostream>
#include <random>
#include <fphistogram/fphistogram.hpp>

template <class Mode>
const char* get_mode_string();
template <> const char* get_mode_string<mtk::fphistogram::mode_log10>() {return "mode_log10";}
template <> const char* get_mode_string<mtk::fphistogram::mode_log2 >() {return "mode_log2" ;}

enum print_mode {
	p_abs = 0,
	p_signed = 1
};

const char* get_print_mode_string(const print_mode mode) {
	switch (mode) {
		case print_mode::p_abs:
			return "abs";
		case print_mode::p_signed:
			return "signed";
		default:
			return "unknown";
	}
}

template <class T, class Mode>
void print_histogram_sw(const print_mode mode, const std::vector<T>& vec) {
	if (mode == print_mode::p_abs) {
		mtk::fphistogram::print_histogram<T, Mode>(vec);
	} else {
		mtk::fphistogram::print_histogram_pm<T, Mode>(vec);
	}
}

template <class T, class Mode>
void print_histogram_sw(const print_mode mode, const T* const array, const std::size_t n) {
	if (mode == print_mode::p_abs) {
		mtk::fphistogram::print_histogram<T, Mode>(array, n);
	} else {
		mtk::fphistogram::print_histogram_pm<T, Mode>(array, n);
	}
}

template <class T, class Mode>
void print_histogram_sw(const print_mode mode, std::function<T(const std::size_t)> iter, std::size_t n) {
	if (mode == print_mode::p_abs) {
		mtk::fphistogram::print_histogram<T, Mode>(iter, n);
	} else {
		mtk::fphistogram::print_histogram_pm<T, Mode>(iter, n);
	}
}

template <class Mode>
void test_random_vec(const print_mode mode) {
	std::printf("# %s\n", __func__);
	std::printf("Log  Mode : %s\n", get_mode_string<Mode>());
	std::printf("PrintMode : %s\n", get_print_mode_string(mode));
	std::mt19937 mt(std::random_device{}());
	std::uniform_real_distribution<double> dist(-1., 1.);

	constexpr std::size_t N = 1lu << 16;
	double fp_list[N];
	for (std::size_t i = 0; i < N; i++) {
		fp_list[i] = dist(mt);
	}
	print_histogram_sw<double, Mode>(mode, fp_list, N);
}

template <class Mode>
void test_random_diff_vec(const print_mode mode) {
	std::printf("# %s\n", __func__);
	std::printf("Log  Mode : %s\n", get_mode_string<Mode>());
	std::printf("PrintMode : %s\n", get_print_mode_string(mode));
	std::mt19937 mt(std::random_device{}());
	std::uniform_real_distribution<double> dist(-1., 1.);

	constexpr std::size_t N = 1lu << 16;
	double fp_list_a[N];
	double fp_list_b[N];
	for (std::size_t i = 0; i < N; i++) {
		fp_list_a[i] = dist(mt);
		fp_list_b[i] = dist(mt);
	}
	std::function<double(const std::size_t)> iter = [&fp_list_a, &fp_list_b](const std::size_t i) {return fp_list_a[i] - fp_list_b[i];};
	print_histogram_sw<double, Mode>(mode, iter, N);
}

template <class Mode>
void test_all_zero(const print_mode mode) {
	std::printf("# %s\n", __func__);
	std::printf("Log  Mode : %s\n", get_mode_string<Mode>());
	std::printf("PrintMode : %s\n", get_print_mode_string(mode));
	constexpr std::size_t n = 1lu << 16;
	double fp_list[n];
	for (std::size_t i = 0; i < n; i++) {
		fp_list[i] = 0.;
	}
	print_histogram_sw<double, Mode>(mode, fp_list, n);
}

template <class Mode>
void test_size_zero(const print_mode mode) {
	std::printf("# %s\n", __func__);
	std::printf("Log  Mode : %s\n", get_mode_string<Mode>());
	std::printf("PrintMode : %s\n", get_print_mode_string(mode));
	std::vector<double> vec;
	print_histogram_sw<double, Mode>(mode, vec);
}

template <class Mode>
void test_half_half(const print_mode mode) {
	std::printf("# %s\n", __func__);
	std::printf("Log  Mode : %s\n", get_mode_string<Mode>());
	std::printf("PrintMode : %s\n", get_print_mode_string(mode));
	constexpr std::size_t n = 1lu << 16;
	double fp_list[n];
	std::size_t i = 0;
	for (; i < n / 2; i++) {
		fp_list[i] = 10;
	}
	for (; i < n; i++) {
		fp_list[i] = 7.;
	}
	print_histogram_sw<double, Mode>(mode, fp_list, n);
}

int main() {
	test_size_zero<mtk::fphistogram::mode_log10>      (print_mode::p_abs   );
	test_size_zero<mtk::fphistogram::mode_log2 >      (print_mode::p_abs   );
	test_all_zero<mtk::fphistogram::mode_log10>       (print_mode::p_abs   );
	test_all_zero<mtk::fphistogram::mode_log2 >       (print_mode::p_abs   );
	test_random_vec<mtk::fphistogram::mode_log10>     (print_mode::p_abs   );
	test_random_vec<mtk::fphistogram::mode_log2 >     (print_mode::p_abs   );
	test_random_diff_vec<mtk::fphistogram::mode_log10>(print_mode::p_abs   );
	test_random_diff_vec<mtk::fphistogram::mode_log2 >(print_mode::p_abs   );
	test_half_half<mtk::fphistogram::mode_log10>      (print_mode::p_abs   );
	test_half_half<mtk::fphistogram::mode_log2 >      (print_mode::p_abs   );

	test_size_zero<mtk::fphistogram::mode_log10>      (print_mode::p_signed);
	test_size_zero<mtk::fphistogram::mode_log2 >      (print_mode::p_signed);
	test_all_zero<mtk::fphistogram::mode_log10>       (print_mode::p_signed);
	test_all_zero<mtk::fphistogram::mode_log2 >       (print_mode::p_signed);
	test_random_vec<mtk::fphistogram::mode_log10>     (print_mode::p_signed);
	test_random_vec<mtk::fphistogram::mode_log2 >     (print_mode::p_signed);
	test_random_diff_vec<mtk::fphistogram::mode_log10>(print_mode::p_signed);
	test_random_diff_vec<mtk::fphistogram::mode_log2 >(print_mode::p_signed);
	test_half_half<mtk::fphistogram::mode_log10>      (print_mode::p_signed);
	test_half_half<mtk::fphistogram::mode_log2 >      (print_mode::p_signed);
}
