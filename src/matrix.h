
/*
 * This library aims to translate gsl functions into ready to use functions inside maxent
 * wator simulator.
 */

#ifndef MATRIX_MAXENT
#define MATRIX_MAXENT


/*
 * The following function computes the inverse of the matrix.
 * matrix data must be provided as a sliced array, to be accessed as
 * matrix[N*i + j] => matrix[i , j]     of dimension N x N
 * the result will be written in the (preallocated) inverse array pointer.
 */ 
void InvertMatrix(const double * matrix, int N, double * inverse) ;


/*
 * The following function computes the determinat of the matrix (N x N)
 * given in input (strided as above).
 * It uses the LU decomposition, so it does not work with singular matrices.
 */
double Determinant(const double * input, int N) ;


/*
 * The following functions diagonalizes the input matrix (N x N)
 * and stores the eigenvalues and eigenvectors in the last two arrays,
 * that must be already initialized.
 * eigen_vect has size NxN, while eigen_val N.
 *
 * The Verbose methods print on stdout the result of the diagonalization, while
 * verbose Diagnoalization calls both.
 */
void Diagonalize(const double * input, int N, double * eigen_val, double * eigen_vect) ;
void VerboseDiagResult(const double * eigen_val, const double * eigen_vect, int N) ;
void VerboseDiagonalization(const double * input, int N, double * eigen_val, double * eigen_vect) ;

/*
 * The following function computes the inner product between a matrix and two vectors.
 * In the Dirac notation it is represented as follows:
 * <v_1 | M | v_2>
 * The matrix, as the method above, is a slice matrix of dimension N x N.
 * Vectors have both dimension N.
 */
double InnerProduct(const double * v_1, const double * matrix, const double * v_2, int N) ;


/*
 * In wator the observables computed by standard methods are linearly dependent.
 * This methods gets rid of the dependences and returns a N - 2 vector of linearly
 * independent quantities.
 */
//void PolishFromIndependent(int q, const double * input, double * output) ;


/*
 * Eliminate from the input array all the indiced contained into fixed_indices.
 */
void ReduceArray(const double * input, double * output, int N_total, int N_fixed, const int * fixed_indices) ;


/*
 * Reduce the rows and columns contained into fixed_indices from input matrix,
 * and stores the result inside output.
 * The matrix in input is stretched, so the elements can be accessed as input[N_total*i + j]
 */
void ReduceStretchedMatrix(const double * input, double * output, int N_total, int N_fixed, const int * fixed_indices) ;


/*
 * This does the same as the one above but the input and output matrix is a 2-rank array.
 */
void Reduce2RankMatrix(const double ** input, double ** output, int N_total, int N_fixed, const int * fixed_indices) ;


/*
 * The following two methods transpose the input matrix, respectively overwriting or not
 * the input
 */
void Transpose(double * input_output, int N) ;
void CopyTranspose(const double * input, double * output, int N) ;

/*
 * Execute the Rows dot Columns multiplication between m1 and m2 matrices (N x N),
 * and stores the results on output. Both the input matrices must be stretched.
 * m[N*i + j] where i varies the row index and j the column index
 */
void MatrixMultiplication(const double * m1, const double * m2, double * output, int N) ;

/*
 * Changes the basis of the input matrix using the unitary U matrix, and stores the results in 
 * output.
 * S_D= U^T S U
 * U passes from the eigenvector basis -> to the canonical basis
 * U^t passes from the canonical basis -> to the eigevector basis.
 */
void MatrixChangeBasis(const double * U, const double * input, double * output, int N) ;


/*
 * Perform the Rows dot Column multiplication between M matrix (N x N) and v vector (N).
 * The result is stored in the output vector.
 */
void MatrixVectorMultiplication(const double * M, const double * v, double * output, int N) ;

#endif
