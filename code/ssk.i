%module ssk
%{
extern double ssk(char* s, char* t, int n, double lambda);
extern double* sskUpTo(char* s, char* t, int n, double lambda);

%}


%typemap(out) double* {
  int i;
  $result = PyList_New(15);
  for (i = 0; i < 15; i++) {
    PyObject *o = PyFloat_FromDouble((double) $1[i]);
    PyList_SetItem($result,i,o);
  }
}


extern double ssk(char* s, char* t, int n, double lambda);
extern double* sskUpTo(char* s, char* t, int n, double lambda);


