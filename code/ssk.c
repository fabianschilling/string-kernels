#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

int count = 0;

double ssk(const char* s, const char* t, const int n, const double lambda);
double K(const char* s, const char* t, const int n, const double lambda);

double ssk(const char* s, const char* t, const int n, const double lambda) {
  printf("SSK called %d\n", ++count);
  double k1 = K(s, s, n, lambda);
  double k2 = K(t, t, n, lambda);
  double k = K(s, t, n, lambda);
  return k / sqrt(k1 * k2);
}

double K(const char* s, const char* t, const int n, const double lambda) {
  double sum = 0.0;
  const int slen = strlen(s);
  const int tlen = strlen(t);
  double ***kp =  (double ***) malloc(2 * sizeof(double **));
  for (int i = 0; i < 2; ++i) {
    kp[i] = (double **) malloc((slen + 1) * sizeof(double *));
    for (int j = 0; j < slen + 1; ++j) {
      kp[i][j] = (double *) malloc((tlen + 1) * sizeof(double));
    }
  }
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

  for (i = 0; i < 2; ++i) {
    for (j = 0; j < (slen + 1); ++j) {
      free(kp[i][j]);
    }
    free(kp[i]);
  }
  free(kp);

  return sum;
}

int main(int argc, char** argv) {
  char s[] = "mar mar";
  char t[] = "hello world";

  printf("%f\n", ssk(s, t, 14, 0.5));
  return 0;
}

