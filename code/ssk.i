%module ssk
%{
extern double ssk(char* s, char* t, int n, double lambda);
%}

extern double ssk(char* s, char* t, int n, double lambda);

