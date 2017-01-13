#include <iostream>
#include <string>
#include <cmath>

double ssk(std::string s, std::string t, int n, double lambda);
double K(std::string s, std::string t, int n, double lambda);

double ssk(std::string s, std::string t, int n, double lambda) {
  double k1 = K(s, s, n, lambda);
  double k2 = K(t, t, n, lambda);
  double k = K(s, t, n, lambda);
  return k / sqrt(k1 * k2);
}

double K(std::string s, std::string t, int n, double lambda) {
  double sum = 0.0;
  double kp[2][s.size() + 1][t.size() + 1];
  int m, i, j;
  double kpp;

  for (m = 0; m < 2; ++m) {
    for (i = 0; i < s.size() + 1; ++i) {
      for (j = 0; j < t.size() + 1; ++j) {
        kp[m][i][j] = (m + 1) % 2;
      }
    }
  }

  for (i = 1; i < n; ++i) {
    for (j = i - 1; j < s.size(); ++j) {
      kp[i % 2][j][i - 1] = 0.0;
    }
    for (j = i - 1; j < t.size(); ++j) {
      kp[i % 2][i - 1][j] = 0.0;
    }
    for (j = i; j < s.size(); ++j) {
      kpp = 0.0;
      for (m = i; m < t.size(); ++m) {
        if (s[j - 1] != t[m - 1]) {
          kpp = lambda * kpp;
        } else {
          kpp = lambda * (kpp + (lambda * kp[(i + 1) % 2][j - 1][m - 1]));
        }
        kp[i % 2][j][m] = lambda * kp[i % 2][j - 1][m] + kpp;
      }
    }
  }

  for (i = n; i < s.size() + 1; ++i) {
    for (j = n; j < t.size() + 1; ++j) {
      if (s[i - 1] == t[j - 1]) {
        sum += lambda * lambda * kp[(n - 1) % 2][i - 1][j - 1];
      }
    }
  }

  return sum;
}

int main(int argc, char** argv) {
  std::string s = "science is organized knowledge";
  std::string t = "wisdom is organized life";

  std::cout << ssk(s, t, 3, 0.5) << std::endl;
  return 0;
}

