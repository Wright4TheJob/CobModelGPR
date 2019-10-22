"""Read a comma or tab-separated text file and perform linear regression."""


def read_file(filename):
    """Read two column contents of file as floats."""
    import csv
    delimiter = "\t"
    xs = []
    ys = []
    with open(filename, 'r') as fin:
        next(fin)  # skip headings
        if delimiter == ',':
            reader = csv.reader(fin)
        else:
            reader = csv.reader(fin, delimiter=delimiter)

        for line in reader:
            xs.append(read_float(0, line))
            ys.append(read_float(1, line))
    return (xs, ys)


def read_float(index, to_read):
    if index is None:
        return None
    else:
        try:
            return float(to_read[index])
        except ValueError:
            print('Float conversion error')


def dot(K, L):
    if len(K) != len(L):
        return 0

    return sum(i[0] * i[1] for i in zip(K, L))


def linear_regression(x_list, y_list):
    """Perform regression analysis on array of x values an list of y values"""
    import numpy as np
    from numpy import dot
    from numpy.linalg import inv
    from scipy import stats
    import math
    n = len(y_list)
    x_array = np.atleast_2d(np.array([np.ones(n), x_list])).transpose()

    # ######### Calculate B values ############
    # Solve for B matrix through pseudo-inverse
    yArray = np.array(y_list).transpose()

    XTX = dot(x_array.transpose(), x_array)
    #  print("X transpose * X = " + repr(XTX))

    XTy = dot(x_array.transpose(), yArray)
    #  print("X transpose * y" + repr(XTy))
    #  print(XTX)
    invXTX = inv(XTX)
    #  print("inv(X transpose * X) = " + repr(invXTX))

    B = dot(invXTX, XTy)

    #  print("Beta Array (Estimated Parameters) = " + repr(B))

    dof = len(yArray) - len(B)

    # ############ Calculate checking statistics #############

    yhat = dot(x_array, B)
    #  print("Predicted value of Y = " + repr(yhat))

    residuals = yArray-yhat
    #  print("Residuals (y-yhat) = " + repr(residuals))

    SSE = dot(residuals.transpose(), residuals)
    SSE = SSE
    #  print("Sum of Squares of the Residuals, SSE = " + repr(SSE))

    residualVariance = SSE/dof

    parameterCOV = invXTX*residualVariance
    #  print("Parameter Covariance = " + repr(parameterCOV))

    SSR = (dot(dot(B.transpose(), x_array.transpose()), yArray)
           - sum(yArray)**2/n)
    SSR = SSR
    #  print("Sum of Squares of the Regression, SSR = " + repr(SSR))

    SST = dot(yArray.transpose(), yArray) - sum(yArray)**2/n
    SST = SST
    #  print("Total sum of squares = " + repr(SST))

    dofSST = n - 1
    dofSSR = len(B)
    dofSSE = dofSST - dofSSR

    p = len(B)
    SigmaSquared = SSE/(n-p)
    #  print('sigma^2$ = %3.4f' % (SigmaSquared))
    # ########### Hypothesis Test Beta terms ###############
    alpha = 1-0.95

    rsquared = 1 - SSE/SST
    print("R Squared = %1.5f" % (rsquared))

    Sxx = []
    for i in range(0, n):
        Sxx.append(sum((x_array[i, :] - np.mean(x_array[i, :]))**2))

    #  print("MSE = " + repr(MSE))
    #  print("Sxx = " + repr(Sxx))
    t0 = [100.1]
    for i in range(1, len(B)):
        t0.append(B[i]/math.sqrt(SigmaSquared*invXTX[i, i]))

    #  print("t0 values for Beta terms = " + repr(t0))

    # reject null hypothesis if |tValue| > t(alpha/2,n-1)
    # Equations from page 310, Chapter 13.3 in (old) book
    tStatistic = stats.distributions.t.ppf(alpha/2, n - 1)
    Bsig = []
    for i in range(1, len(B)):
        if abs(t0[i]) > abs(tStatistic):
            Bsig.append(1)
            #  print("B" + repr(i) + " is significant")
        else:
            Bsig.append(0)
            #  print("B" + repr(i) + " is not significant")

    # ############# Confidence intervals for Beta values ##################

    print("Confidence Interval for Beta Values")
    for i, B1, C in zip(range(len(B)), B, np.diagonal(parameterCOV)):
        lowB = B1 - stats.t.ppf(1-alpha/2, dofSSE) * math.sqrt(C)
        highB = B1 + stats.t.ppf(1-alpha/2, dofSSE) * math.sqrt(C)
        print("B%i = %g" % (i, B1))
        print("%g < B%i < %g with 95%% confidence" % (lowB, i, highB))


def main():
    import sys

    try:
        filename = sys.argv[1]
    except IndexError:
        print('Please provide a file name.')
        return

    (x_list, y_list) = read_file(filename)
    linear_regression(x_list, y_list)


if __name__ == "__main__":
    main()
