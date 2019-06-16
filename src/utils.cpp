#include <cmath>


// Compute the pearson correlation
double pearson_correlation(int n, const double * predictions, const double * real_values,
			   double * error) {
  // Compute the pearson correlati1on coefficient
  double m_pred = 0, m_real = 0;
  double sigma_pred = 0, sigma_real = 0, cov = 0;


  for (int i = 0; i < n; ++i) {
    m_pred += predictions[i];
    m_real += real_values[i];
    sigma_pred += predictions[i] * predictions[i];
    sigma_real += real_values[i] * real_values[i];
    cov += predictions[i] * real_values[i];
  }

  m_pred /= n;
  m_real /= n;
  sigma_pred /= n;
  sigma_real /= n;
  cov /= n;

  double r =  (cov - m_pred * m_real) / sqrt(sigma_pred * sigma_real);

  if (error)
    *error = sqrt( (1- r*r) / (n - 2));

  return r;
}


// Extract gaussian random numbers
double random_normal(double m, double sigma) {
  double X = rand() / (double) RAND_MAX;
  double Y = rand() / (double) RAND_MAX;

  double gauss = sqrt( - 2 * log(1 - Y)) * cos( 2 * M_PI * X);

  return m + gauss * sigma;
}
