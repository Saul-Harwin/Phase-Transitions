import numpy as np
from scipy.stats import chi2 as chi2_dist

def weighted_least_squares_fit(x, y, y_err, deg=2):
    """
    Parameters:
        x:  array-like
            Independent variable data.
        y : array-like
            Dependent variable data.
        y_err : array-like
            Error in the dependent variable data.
        deg : int
            Degree of the polynomial to fit.
    Returns:
        coeffs : ndarray
            Coefficients of the fitted polynomial, highest degree first.
        cov : ndarray
            Covariance matrix of the polynomial coefficients. This allows you to propagate uncertainties to derived quantities (e.g. the location of a maximum).
    """
    
    if y_err is not None:
        w = 1.0 / (y_err)
    else:
        w = None
    
    # Weighted polynomial fit
    coeffs, cov = np.polyfit(x, y, deg=deg, w=w, cov=True)

    # Polynomial object
    p = np.poly1d(coeffs)

    # Goodness of fit
    residuals = y - p(x)
    
    if y_err is not None:
        chi2_val = np.sum((residuals / y_err)**2)
    else:
        chi2_val = np.sum(residuals**2)
    
    dof = len(y) - (deg + 1)
    chi2_red = chi2_val / dof

    # Correct p-value calculation
    p_value = 1.0 - chi2_dist.cdf(chi2_val, dof)

    return p, cov, chi2_red, p_value