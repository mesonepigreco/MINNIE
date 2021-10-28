#include "matrix.h"
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>
#include <stdio.h>
#include <gsl/gsl_eigen.h>
/*
 * This library uses the GSL gnu library 
 * to operate with matrices
 */


/* 
 * Invert the matrix and store the result into inverse (already allocated)
 * data stores as matrix[N*i + j], of dimension N*N.
 */
void InvertMatrix( double * matrix, int N, double * inverse) {
  int i, j, s;
  gsl_matrix_view m, inv;
  gsl_permutation *p;
  double * lu_matrix;

  // Copy the original matrix for the LU decomposition
  lu_matrix = (double *) calloc(sizeof(double), N*N);
  for (i = 0; i < N*N; ++i) lu_matrix[i] = matrix[i];


  // Load the matrix as gsl struct
  m = gsl_matrix_view_array(lu_matrix, N, N);
  inv = gsl_matrix_view_array(inverse, N,N);
  p = gsl_permutation_alloc(N);

  // Perform the LU decomposition and inversion
  gsl_linalg_LU_decomp(&m.matrix, p, &s);  
  gsl_linalg_LU_invert(&m.matrix, p, &inv.matrix);

  // Free
  gsl_permutation_free(p);
  free(lu_matrix);
}


// det A
double Determinant( double * input, int N) {
  int i, s;
  gsl_matrix_view m;
  gsl_permutation *p;
  double * lu_matrix;
  double det;

  // Copy the original matrix to get the LU decomposiotion
  lu_matrix = (double *) malloc(sizeof(double) * N * N);
  for (i = 0; i < N*N; ++i) lu_matrix[i] = input[i];

  // Load the matrix as gsl struct
  m = gsl_matrix_view_array(lu_matrix, N, N);
  p = gsl_permutation_alloc(N);

  gsl_linalg_LU_decomp(&m.matrix, p, &s);
  det = gsl_linalg_LU_det(&m.matrix, s);
  
  free (lu_matrix);
  gsl_permutation_free(p);

  return det;
}


void Diagonalize( double * input, int N, double * eigen_val, double * eigen_vect) {
  double * copy;
  int i;
  gsl_eigen_symmv_workspace * w;
  gsl_vector_view eigval;
  gsl_matrix_view m, eigvect;

  copy = (double*) malloc(sizeof(double)*N*N);
  for (i = 0; i < N*N; ++i) copy[i] = input[i];

  m = gsl_matrix_view_array(copy, N, N);
  eigval = gsl_vector_view_array(eigen_val, N);
  eigvect = gsl_matrix_view_array(eigen_vect, N, N);
  
  // Allocate the workspace
  w = gsl_eigen_symmv_alloc(N);

  // Compute eigenvalues and eigenvectors
  gsl_eigen_symmv(&m.matrix, &eigval.vector, &eigvect.matrix, w);

  // Sort eigenvalue and eigen vector 
  gsl_eigen_symmv_sort(&eigval.vector, &eigvect.matrix, GSL_EIGEN_SORT_ABS_DESC);

  // Free memory
  gsl_eigen_symmv_free(w);
  free(copy);
}


void VerboseDiagResult( double * eigen_val,  double * eigen_vect, int N) {
  int i, j;
  printf("\nEigenvalues:\n");
  for (i = 0; i < N; ++i) {
    printf("\t%10.5e\n", eigen_val[i]);
  }

  printf("\nEigenvectors (columns):\n");
  for (i = 0; i < N; ++i) {
    for (j = 0; j < N; ++j) {
      printf("\t%10.5e", eigen_vect[N*i + j]);
    }
    printf("\n");
  }
}


void VerboseDiagonalization(double * input, int N, double * eigen_val, double * eigen_vect) {
  Diagonalize(input, N, eigen_val, eigen_vect);
  VerboseDiagResult(eigen_val, eigen_vect, N);
}


// computes <v_1 | M | v_2>
double InnerProduct(double * v_1, double * matrix, double * v_2, int N) {
  gsl_matrix_view m;
  gsl_vector_view v1, v2;
  gsl_vector * tmp;
  double inner_prod;

  // Initializing the GSL representation
  m = gsl_matrix_view_array(matrix, N, N);
  v1 = gsl_vector_view_array(v_1, N);
  v2 = gsl_vector_view_array(v_2, N);

  tmp = gsl_vector_alloc(N);
  
  // matrix \dot v_2 => tmp
  gsl_blas_dgemv(CblasNoTrans, 1.0, &m.matrix, &v2.vector, 0, tmp);

  // Print medium
  printf("\n");
  int i;
  for (i = 0; i < N; ++i) printf(" %g ", gsl_vector_get(tmp, i));
  printf("\n");

  // v_1 \dot tmp => inner_prod
  gsl_blas_ddot(&v1.vector, tmp, &inner_prod);

  // Free memory
  gsl_vector_free(tmp);

  return inner_prod;
}

/*
 * The following method takes the input a dependent array and gives as output an
 * independend array of size N - 2
 */
/* void PolishFromIndependent(int q, const double * input, double * output) {
  int i;
  for (i = 1; i < GetNparamFromQ(q); ++i) {
    if (i < q)
      output[i - 1] = input[i];
    else if (i > q)
      output[i - 2] = input[i];
  }
} */


void ReduceArray( double * input, double * output, int N_total, int N_fixed,  int * fixed_indices) {
  int i, j, jump=0;
  for (i = 0; i < N_total; ++i) {

    // Check if the current index must be jumped
    for (j = 0; j<N_fixed;++j) {
      if (i == j) {
	jump++;
	break;
      }
    }
    
    output[i] = input[i+jump];
  }
}


void ReduceStretchedMatrix( double * input, double * output, int N_total, int N_fixed,  int * fixed_indices) {
  int i, j, k, jump_i=0, jump_j=0;
  for (i = 0; i < N_total; ++i) {
    // Check if the current index must be jumped
    for (k = 0; k<N_fixed;++k) {
      if (i == k) {
	jump_i++;
	break;
      }
    }

    for(j = 0; j < N_total; ++j) {

      // Check if the j index must be jumped
      for (k = 0; k<N_fixed;++k) {
	if (j == k) {
	  jump_j++;
	  break;
	}
      }
    
      output[N_total* i + j] = input[N_total* (i+jump_i) + j + jump_j];
    }
  }
}

void Reduce2RankMatrix( double ** input,  double ** output, int N_total, int N_fixed,  int * fixed_indices) {
  int i, j, jump=0;

  for (i = 0; i < N_total; ++i) {
    
    // Check if the current index must be jumped
    for (j = 0; j<N_fixed;++j) {
      if (i == j) {
	jump++;
	break;
      }
    }
    
    ReduceArray(input[i+jump], output[i], N_total, N_fixed, fixed_indices);
  }
}


void CutLastMValuesMatrixStretched(double * input_output, int N_total, int m) {
  int i;
  int * fixed_indices = (int*) malloc(sizeof(int) * m);
  double * copy = (double *) malloc(sizeof(double) * N_total * N_total);

  for (i = 0; i < m; ++i) fixed_indices[i] = N_total - m - 1;
  ReduceStretchedMatrix(input_output, copy, N_total, m, fixed_indices);
  
  for (i = 0; i < N_total*N_total; ++i)  input_output[i] = copy[i];

  free(fixed_indices);
  free(copy);
}

void CutCopyLastMValuesMatrixStretched(const double * input, double * output, int N_total, int m) {
  int i;
  for (i = 0; i < N_total * N_total; ++i) output[i] = input[i];
  CutLastMValuesMatrixStretched(output, N_total, m);
}


void Transpose(double * input_output, int N) {
  int i, j;
  double tmp;
  for (i = 0; i <N; ++i) {
    for (j = i + 1; j < N; ++j) {
      tmp = input_output[N*i + j];
      input_output[N*i + j] = input_output[N*j + i];
      input_output[N*j + i] = tmp;
    }
  }
}

void CopyTranspose( double * input, double * output, int N) {
  int i;
  for (i = 0; i < N*N; ++i) output[i] = input[i];
  Transpose(output, N);
}


void MatrixMultiplication( double * m1,  double * m2, double * output, int N) {
  int i, j, k;

  for (i = 0; i < N; ++i) {
    for (j = 0; j < N; ++j) {
      output[N*i + j] = 0;
      for(k = 0; k< N; ++k) {
	output[N*i + j] += m1[N*i + k] * m2[N*k + j];
      }
    }
  }
}



/*
 * if U has eigenvector as columns
 * S_D= U^T S U
 * U passes from the eigenvector basis -> to the canonical basis
 * U^t passes from the canonical basis -> to the eigevector basis.
 */
void MatrixChangeBasis( double * U,  double * input, double * output, int N) {
  int i;
  double * tmp1 = (double *) calloc(sizeof(double*), N*N);
  double * tmp2 = (double *) calloc(sizeof(double*), N*N);
  
  for(i = 0; i < N*N; ++i) output[i] = 0; 
  
  MatrixMultiplication(input, U, tmp1, N);
  CopyTranspose(U, tmp2, N);
  MatrixMultiplication(tmp2, tmp1, output, N);

  free(tmp1);
  free(tmp2);
}


/*
 * Sizes:
 * M = N x N
 * v = N
 * output = N
 */
void MatrixVectorMultiplication( double * M,  double * v, double * output, int N) {
  int i, j;

  for (i = 0; i < N; ++i) {
    output[i] = 0;
    for (j = 0; j < N; ++j) {
      output[i] += M[N*i + j] * v[j];
    }
  }
}
