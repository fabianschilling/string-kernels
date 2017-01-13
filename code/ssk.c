#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

int count = 0;
int upTo = 0;
double Ki[15] = {1};
double Kiss[15] = {1};
double Kitt[15] = {1};

double ssk(const char* s, const char* t, const int n, const double lambda);
double K(const char* s, const char* t, const int n, const double lambda, double* Ki);

double ssk(const char* s, const char* t, const int n, const double lambda) {
  printf("SSK called %d\n", ++count);
  double k1 = K(s, s, n, lambda);
  double k2 = K(t, t, n, lambda);
  double k = K(s, t, n, lambda);
  return k / sqrt(k1 * k2);
}

double* sskUpTo(const char* s, const char* t, const int n, const double lambda) {
  //printf("start\n");
  upTo = 1;
  double k1 = K(s, s, n, lambda, Kiss);
  double k2 = K(t, t, n, lambda, Kitt);
  double k = K(s, t, n, lambda, Ki);
  //printf("mid\n");
  for(int i = 0; i < 15; i++)
  {
    //printf("Ki: %d\n", i);
    Ki[i] = Ki[i] / sqrt(Kiss[i] * Kitt[i]);
    //printf("post\n");
  }
  //printf("b4return\n");
  return Ki;
}

double K(const char* s, const char* t, const int n, const double lambda, double* Ki) {
  double sum = 0.0;
  double sumi = 0.0;
  const int slen = strlen(s);
  const int tlen = strlen(t);

  //printf("premalloc\n");

  double ***kp =  (double ***) malloc(2 * sizeof(double **));
  for (int i = 0; i < 2; ++i) {
    kp[i] = (double **) malloc((slen + 1) * sizeof(double *));
    for (int j = 0; j < slen + 1; ++j) {
      kp[i][j] = (double *) malloc((tlen + 1) * sizeof(double));
    }
  }
  //printf("postmalloc\n");
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
      if((i-1) < (tlen + 1))
        kp[i % 2][j][i - 1] = 0.0;
    }
    for (j = i - 1; j < tlen; ++j) {
      if((i-1) < (slen + 1))
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
        if(upTo)
	{
          //printf("in upto %d\n", i);
          sumi = 0;
	  for (int ii = i; ii < slen + 1; ++ii) {
	    for (int ji = i; ji < tlen + 1; ++ji) {
	      if (s[ii - 1] == t[ji - 1]) {
		sumi += lambda * lambda * kp[(i - 1) % 2][ii - 1][ji - 1];
	      }
	    }
	  }
          Ki[i] = sumi;
          //printf("out of upto\n");
	}
  }

  //printf("calculating last\n");
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
      //printf("freed j:%d/%d n=%d\n", j,slen,n);
    }
    free(kp[i]);
  }
  free(kp);
  //printf("Kpfree\n");
  Ki[n] = sum;
  //printf("Kreturn\n");
  return sum;
}

int main(int argc, char** argv) {
  char s[] = "mar mar";
  char t[] = "hello world";

  printf("%f\n", ssk(s, t, 14, 0.5));
  return 0;
}

