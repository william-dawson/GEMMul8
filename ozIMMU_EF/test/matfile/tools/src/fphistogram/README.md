# FP Histogram

This library draws an exponent histogram of a given floating point array.

## Sample code
```cpp
// sample.cpp
// gcc -I/path/to/fphistogram/include -std=c++11 sample.cpp [Optional:-fopenmp]
#include <random>
#include <fphistogram/fphistogram.hpp>

int main() {
	std::mt19937 mt(std::random_device{}());
	std::uniform_real_distribution<double> dist(-10., 10.);

	constexpr std::size_t N = 1lu << 16;
	double fp_list[N];
	for (std::size_t i = 0; i < N; i++) {
		fp_list[i] = dist(mt);
	}
	mtk::fphistogram::print_histogram(fp_list, N);

	// std::vector<float> fp_list_vec;
	// mtk::fphistogram::print_histogram(fp_list_vec);

	// std::function<double(const std::size_t)> iter = [&fp_list](const std::size_t i) {return 3.0 * fp_list[i];};
	// mtk::fphistogram::print_histogram(iter, N);
}
```

Then you can get a histogram like this.
```
[  exp ](   count  ){    ratio   }
[     3](     13099){1.998749e-01}:*******************
[     2](     26306){4.013977e-01}:****************************************
[     1](     13145){2.005768e-01}:********************
[     0](      6446){9.835815e-02}:*********
[    -1](      3296){5.029297e-02}:*****
[    -2](      1599){2.439880e-02}:**
[    -3](       808){1.232910e-02}:*
[    -4](       438){6.683350e-03}:
[    -5](       209){3.189087e-03}:
[    -6](        98){1.495361e-03}:
[    -7](        40){6.103516e-04}:
[    -8](        24){3.662109e-04}:
[    -9](        14){2.136230e-04}:
[   -10](         6){9.155273e-05}:
[   -11](         3){4.577637e-05}:
[   -12](         2){3.051758e-05}:
[   -13](         3){4.577637e-05}:
-----
[ zero ](         0){0.000000e+00}:
```

- `pm` mode.

```cpp
mtk::fphistogram::print_histogram_pm(fp_list, N);
```

Output:
```
                         (  -count  ){   -ratio   }[  exp ](  +count  ){   +ratio   }
 ************************(     16375){2.498627e-01}:[   +0](     16420){2.505493e-01}:*************************
             ************(      8224){1.254883e-01}:[   -1](      8147){1.243134e-01}:************
                   ******(      4159){6.346130e-02}:[   -2](      4002){6.106567e-02}:******
                      ***(      1979){3.019714e-02}:[   -3](      2073){3.163147e-02}:***
                        *(      1037){1.582336e-02}:[   -4](      1089){1.661682e-02}:*
                         (       494){7.537842e-03}:[   -5](       503){7.675171e-03}:
                         (       244){3.723145e-03}:[   -6](       286){4.364014e-03}:
                         (       122){1.861572e-03}:[   -7](       121){1.846313e-03}:
                         (        69){1.052856e-03}:[   -8](        60){9.155273e-04}:
                         (        35){5.340576e-04}:[   -9](        29){4.425049e-04}:
                         (        16){2.441406e-04}:[  -10](        19){2.899170e-04}:
                         (        10){1.525879e-04}:[  -11](         7){1.068115e-04}:
                         (         3){4.577637e-05}:[  -12](         3){4.577637e-05}:
                         (         2){3.051758e-05}:[  -13](         4){6.103516e-05}:
                         (         0){0.000000e+00}:[  -14](         2){3.051758e-05}:
                         (         1){1.525879e-05}:[  -15](         0){0.000000e+00}:
                         (         0){0.000000e+00}:[  -16](         1){1.525879e-05}:
```

## Requirements
- C++ >= 11


## License

MIT
