import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Extract data from original csv and convert them into arrays based on their headers
def process_file2(file) -> tuple:
    """Assign multiple arrays named year, month, decimal_date, ave_co2, dseasonalized, ndays, sdev, uncertainty.
    """

    year, month, decimal_date, ave_co2, dseasonalized, ndays, sdev, uncertainty = np.loadtxt(
        file, comments='#', delimiter=',', unpack=True, skiprows=42)
    return year, month, decimal_date, ave_co2, dseasonalized, ndays, sdev, uncertainty


# Remove invalid datapoints
def remove_invalid_pts(year, month, decimal_date, ave_co2, dseasonalized, ndays, sdev, uncertainty):
    """Remove invalid datapoints, which is defined by negative uncertainty."""

    valid_indices = uncertainty > 0
    return year[valid_indices], month[valid_indices], decimal_date[valid_indices], ave_co2[valid_indices], \
           dseasonalized[valid_indices], ndays[valid_indices], sdev[valid_indices], uncertainty[valid_indices]


# Periodic Function
def periodic_func(t, a, b, c, phi) -> float:
    """Return the periodic function for linear growth and periodicity."""

    return a + b * t + c * np.sin(2 * np.pi * t - phi)

# Get parameters and covariance matrix
def params_n_cov(file) -> tuple:
    """Return the coefficients for the periodic function (named params) and the covariance matrix.
    """

    year, month, decimal_date, ave_co2, dseasonalized, ndays, sdev, uncertainty = process_file2(file)
    params, covariance = curve_fit(periodic_func, decimal_date, ave_co2, p0=[300, 0.1, 2.5, 0], sigma=uncertainty, absolute_sigma=True)
    return params, covariance


# Find residuals
def find_residual(decimal_date, ave_co2, uncertainty):
    """Return the residuals and the coefficients.
    """

    params, covariance = curve_fit(periodic_func, decimal_date, ave_co2, p0=[300, 0.1, 2.5, 0], sigma=uncertainty, absolute_sigma=True)
    residuals = ave_co2 - periodic_func(decimal_date, *params)
    return residuals, params


# Dataset | Errorbar plot
def plot(decimal_date, ave_co2, uncertainty, params):
    """Return a plot of data points and errorbars.
    """

    plt.errorbar(decimal_date, ave_co2, yerr=uncertainty, fmt='o', label='Data', markersize=2
                 )
    plt.plot(decimal_date, periodic_func(decimal_date, *params), label='Fitted Model', color='red')
    plt.title('Decimal Date vs. Average CO2 Concentration (ppm)')
    plt.xlabel('Decimal Date')
    plt.ylabel('Average CO2 Concentration (ppm)')
    plt.legend()
    plt.show()


# Residual Plot
def plot_residuals(decimal_date, residuals, uncertainty):
    """Return a residual plot.
    """

    plt.errorbar(decimal_date, residuals, yerr=uncertainty, fmt='o', label='Residuals', markersize=4)
    plt.axhline(0, color='red', linestyle='--', label='Zero Residual Line')
    plt.xlabel('Decimal Date')
    plt.ylabel('Residuals (ppm)')
    plt.title('Residuals of the Fitted Model')
    plt.legend()
    plt.show()


def main():
    # Load and clean data
    year, month, decimal_date, ave_co2, dseasonalized, ndays, sdev, uncertainty = process_file2('co2_mm_mlo.csv')
    year, month, decimal_date, ave_co2, dseasonalized, ndays, sdev, uncertainty = remove_invalid_pts(year, month, decimal_date, ave_co2, dseasonalized, ndays, sdev, uncertainty)

    # Fit the model and get parameters
    params, covariance = params_n_cov('co2_mm_mlo.csv')
    print(f"Parameters: {params}")
    print(f"Covariance Matrix: {covariance}")

    # Compute residuals
    residuals, params = find_residual(decimal_date, ave_co2, uncertainty)
    print(f"Residuals: {residuals}")

    # Plot the data and the model fit
    plot(decimal_date, ave_co2, uncertainty, params)

    # Plot the residuals
    plot_residuals(decimal_date, residuals, uncertainty)

if __name__ == '__main__':
    main()
