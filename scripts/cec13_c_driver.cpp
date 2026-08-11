/* Linux validation driver around the OFFICIAL CEC'13 test_func (test_func.cpp).
 *
 * Reads a points file:  first line "<npoints> <dim>", then npoints rows of dim
 * doubles. Evaluates every one of the 28 functions at every point and prints
 *     <func_num> <point_index> <value>
 * one per line. Compile together with a Linux-patched copy of test_func.cpp
 * (see scripts/validate_vs_c.py, which removes <WINDOWS.H> and fixes the
 * %Lf -> %lf fscanf bug). Run with cwd = the C code dir so its relative
 * "input_data/..." resolves.
 */
#include <stdio.h>
#include <stdlib.h>

void test_func(double *, double *, int, int, int);

/* Globals test_func.cpp declares `extern`; defined once here (as main.cpp did).
 * Zero-initialised, so test_func's first-call free() on them is a safe free(NULL). */
double *OShift, *M, *y, *z, *x_bound;
int ini_flag = 0, n_flag, func_flag;

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s points.txt\n", argv[0]); return 1; }
    FILE *fp = fopen(argv[1], "r");
    if (!fp) { fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }

    int npts = 0, dim = 0;
    if (fscanf(fp, "%d %d", &npts, &dim) != 2) {
        fprintf(stderr, "bad header (expected '<npoints> <dim>')\n"); return 1;
    }
    double *x = (double *)malloc(sizeof(double) * npts * dim);
    for (int i = 0; i < npts * dim; i++)
        if (fscanf(fp, "%lf", &x[i]) != 1) {
            fprintf(stderr, "bad data value at index %d\n", i); return 1;
        }
    fclose(fp);

    double *f = (double *)malloc(sizeof(double) * npts);
    for (int fn = 1; fn <= 28; fn++) {
        test_func(x, f, dim, npts, fn);      /* all npts points for this function */
        for (int p = 0; p < npts; p++)
            printf("%d %d %.12e\n", fn, p, f[p]);
    }
    free(x); free(f);
    return 0;
}
