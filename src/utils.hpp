#ifndef UTILS_NN
#define UTILS_NN

/*
 * This library contains useful method
 */


/*
 * This function evaluates the pearson correlation coefficient between
 * the two variables v1 and v2, of size N.
 * If given, the last argument is filled with the expected error.
 */
double pearson_correlation(int N, const double * v1,
			   const double * v2, double * error = NULL) ;




#endif
