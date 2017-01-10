#include <stdio.h>
#include <math.h>
#include <string.h>

double ssk(char* s, char* t, int n, double lambda);
double K(char* s, char* t, int n, double lambda);

double ssk(char* s, char* t, int n, double lambda) {
  double k1 = K(s, s, n, lambda);
  double k2 = K(t, t, n, lambda);
  double k = K(s, t, n, lambda);
  return k / sqrt(k1 * k2);
}

double K(char* s, char* t, int n, double lambda) {
  double sum = 0.0;
  int slen = strlen(s);
  int tlen = strlen(t);
  double kp[2][slen + 1][tlen + 1];
  int m, i, j;
  double kpp;

  for (m = 0; m < 2; ++m) {
    for (i = 0; i < slen + 1; ++i) {
      for (j = 0; j < tlen + 1; ++j) {
        kp[m][i][j] = (m + 1) % 2;
      }
    }
  }

  for (i = 1; i < n; ++i) {
    for (j = i - 1; j < slen; ++j) {
      kp[i % 2][j][i - 1] = 0.0;
    }
    for (j = i - 1; j < tlen; ++j) {
      kp[i % 2][i - 1][j] = 0.0;
    }
    for (j = i; j < slen; ++j) {
      kpp = 0.0;
      for (m = i; m < tlen; ++m) {
        if (s[j - 1] != t[m - 1]) {
          kpp = lambda * kpp;
        } else {
          kpp = lambda * (kpp + (lambda * kp[(i + 1) % 2][j - 1][m - 1]));
        }
        kp[i % 2][j][m] = lambda * kp[i % 2][j - 1][m] + kpp;
      }
    }
  }

  for (i = n; i < slen + 1; ++i) {
    for (j = n; j < tlen + 1; ++j) {
      if (s[i - 1] == t[j - 1]) {
        sum += lambda * lambda * kp[(n - 1) % 2][i - 1][j - 1];
      }
    }
  }

  return sum;
}

int main(int argc, char** argv) {
  char s[] = "science is organized knowledge";
  char t[] = "wisdom is organized life";

  printf("%lf", ssk(s, t, 3, 0.5));
  return 0;
}

