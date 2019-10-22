# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 16:00:27 2015

@author: davidwright
"""


def plotConfidence(xList, yList, xTitle, yTitle, plotTitle):
    import matplotlib.pyplot as plt
    import numpy as np
    from numpy import dot
    from numpy.linalg import inv
    from scipy import stats
    import math

    n = len(yList)

    alpha = 0.05

    ones = np.ones(len(yList))
    xArray = np.array([ones, xList]).transpose()
    yArray = np.array([yList]).transpose()
    XTX = dot(xArray.transpose(), xArray)
    # print("X transpose * X = " + repr(XTX))

    XTy = dot(xArray.transpose(), yArray)
    # print("X transpose * y" + repr(XTy))

    invXTX = inv(XTX)
    # print("inv(X transpose * X) = " + repr(invXTX))

    B = dot(invXTX, XTy)

    xSpacing = np.arange(min(xList), max(xList), 0.1)
    yhat = B[0] + B[1]*xList

    tStatistic = stats.distributions.t.ppf(alpha/2, n - 1)
    xBar = np.mean(xList)

    SSxx = 0
    SSE = 0
    for i in range(0, len(xList)):
        SSxx += (xList[i] - xBar)**2
        SSE += (yList[i] - yhat[i])**2

    yhatPlotting = B[0] + B[1]*xSpacing

    SSquared = SSE/(n-2)
    S = math.sqrt(SSquared)

    yUpperEstimate = []
    yLowerEstimate = []
    yUpperPrediction = []
    yLowerPrediction = []
    for i in range(0, len(xSpacing)):
        yUpperEstimate.append(yhatPlotting[i] +
                              tStatistic*S*math.sqrt(1/n +
                              (xSpacing[i]-xBar)**2/SSxx))
        yLowerEstimate.append(yhatPlotting[i] -
                              tStatistic*S*math.sqrt(1/n +
                              (xSpacing[i]-xBar)**2/SSxx))
        yLowerPrediction.append(yhatPlotting[i] -
                                tStatistic*S*math.sqrt(1+1/n +
                                (xSpacing[i]-xBar)**2/SSxx))
        yUpperPrediction.append(yhatPlotting[i] +
                                tStatistic*S*math.sqrt(1+1/n +
                                (xSpacing[i]-xBar)**2/SSxx))

    modelString = "Linear Model: y = %3.2f + %3.2fx" % (B[0], B[1])

    textX = 0.05*(max(xSpacing) - min(xSpacing)) + min(xSpacing)
    textY = 0.05*(max(yList) - min(yList)) + min(yList)

    fig = plt.figure()
    fig.set_size_inches(6, 4)

    plt.plot(xList, yList, '.', label='Raw Data')
    plt.plot(xSpacing, yhatPlotting, label='Linear Fit')
    plt.plot(xSpacing, yUpperEstimate,
             label='95% Estimation Interval', color='red',
             linestyle='--')
    plt.plot(xSpacing, yLowerEstimate, color='red', linestyle='--')
    plt.plot(xSpacing, yUpperPrediction,
             label='95% Estimation Interval', color='black',
             linestyle='-.')
    plt.plot(xSpacing, yLowerPrediction, color='black',
             linestyle='-.')
    plt.text(textX, textY, modelString, fontsize=12)
    plt.xlabel(xTitle)
    plt.ylabel(yTitle)
    lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.grid()
    plt.savefig('ConfidencePlot-' + plotTitle + '.png', dpi=100,
                bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()
    plt.close()
    return


def twofactorANOVA(treatmentA, treatmentB, yList, AName, BName,
                   replicates=6):
    """Perform two factor ANOVA."""
    import numpy as np
    from scipy.stats import f
    aLevels = np.unique(treatmentA).tolist()
    bLevels = np.unique(treatmentB).tolist()
    a = len(aLevels)
    b = len(bLevels)
    N = len(yList)
    n = replicates

    yidd = np.zeros(a)
    for i in range(0, N):
        aLevel = treatmentA[i]
        yidd[aLevels.index(aLevel)] += yList[i]

    yjdd = np.zeros(b)
    for i in range(0, N):
        bLevel = treatmentB[i]
        yjdd[bLevels.index(bLevel)] += yList[i]

    yijd = np.zeros((a, b))

    for i in range(0, N):
        aLevel = treatmentA[i]
        bLevel = treatmentB[i]
        yijd[aLevels.index(aLevel), bLevels.index(bLevel)] += yList[i]

    GrandTotal = np.sum(yList)

    if a == 0:
        print("Zero levels detected for treatment A")
    if b == 0:
        print("Zero levels detected for treatment B")
    SSA = 1/(b*n)*np.sum(np.square(yidd)) - GrandTotal**2/(a*b*n)
    SSB = 1/(a*n)*np.sum(np.square(yjdd)) - GrandTotal**2/(a*b*n)

    SSSub = 1/n*np.sum(np.sum(np.square(yijd))) - GrandTotal**2/(a*b*n)
    SSAB = SSSub - SSA - SSB
    SST = np.sum(np.square(yList)) - GrandTotal**2/(a*b*n)
    SSE = SST - SSAB - SSA - SSB
    dofA = a-1
    dofB = b-1
    dofAB = (a-1)*(b-1)
    dofE = a*b*(n-1)
    dofT = a*b*n-1
    MSA = SSA/dofA
    MSB = SSB/dofB
    MSAB = SSAB/dofAB
    MSE = SSE/dofE
    f0A = MSA/MSE
    f0B = MSB/MSE
    f0AB = MSAB/MSE

    # Compute associated P-Value for the F0 statistic
    pA = 1-f.cdf(f0A, dofA, dofE)
    pB = 1-f.cdf(f0B, dofB, dofE)
    pAB = 1-f.cdf(f0AB, dofAB, dofE)

    filename = 'Results-ANOVA.txt'
    fout = open(filename, 'w')

    # Print all of the resulting information
    fout.write('Analysis of Variance Table\n')
    fout.write('Source of variation.\tSum of Square\tDOF\t\
               Mean Square\tF_0\tP-Value\n')
    fout.write('%s        \t%0.02f     \t%g \t%0.2f    \t%0.2f\
               \t%0.6f\n' % (AName, SSA, dofA, MSA, f0A, pA))
    fout.write('%s       \t%0.02f     \t%g \t%0.2f    \t%0.2f \
               \t%0.6f\n' % (BName, SSB, dofB, MSB, f0B, pB))
    fout.write('Interaction \t%0.02f \t%g \t%0.2f \t%0.2f \t%0.6f\n'
               % (SSAB, dofAB, MSAB, f0AB, pAB))
    fout.write('Error \t%0.02f   \t%g \t%0.2f\n' % (SSE, dofE, MSE))
    fout.write('Total \t%0.02f   \t%g\n' % (SST, dofT))

    return


def doRegression(xArray, yList, DescriptionString):
    """Perform regression analysis on array of x values an list of y values"""
    import numpy as np
    from numpy import dot
    from numpy.linalg import inv
    from matplotlib import pyplot as plt
    from scipy import stats
    import math
    from scipy.stats import f

    n = len(yList)

    # ######### Calculate B values ############
    filename = 'RegressionResults-' + DescriptionString + '.txt'
    fout = open(filename, 'w')

    # Solve for B matrix through pseudo-inverse
    yArray = np.array(yList).transpose()
    fout.write("Observed Output (y Array):\n")
    for i in range(0, n):
        fout.write(repr(yList[i]) + "\n")

    fout.write("x Array (Design Matrix):\n")
    for i in range(0, n):
        for j in range(0, len(xArray[0, :])):
            fout.write(repr(xArray[i, j]) + "\t")
        fout.write('\n')

    XTX = dot(xArray.transpose(), xArray)
    XTy = dot(xArray.transpose(), yArray)
    invXTX = inv(XTX)

    B = dot(invXTX, XTy)
    fout.write("Beta Array (Estimated Parameters) = " + repr(B)+"\n")
    dof = len(yArray) - len(B)

    # ############ Calculate checking statistics #############
    yhat = dot(xArray, B)
    residuals = yArray-yhat

    SSE = dot(residuals.transpose(), residuals)
    SSE = SSE

    residualVariance = SSE/dof
    parameterCOV = invXTX*residualVariance

    SSR = dot(dot(B.transpose(), xArray.transpose()), yArray) -\
        sum(yArray)**2/n
    SSR = SSR

    SST = dot(yArray.transpose(), yArray) - sum(yArray)**2/n
    SST = SST

    dofSST = n - 1
    dofSSR = len(B)
    dofSSE = dofSST - dofSSR

    MSR = SSR/dofSSR
    MSE = SSE/dofSSE

    p = len(B)
    SigmaSquared = SSE/(n-p)
    fout.write('$\sigma^2$ = %3.4f\n' % (SigmaSquared))
    # Hypothesis Test Beta terms
    Fo = MSR/MSE
    alpha = 1-0.95

    Fcrit = stats.f.ppf(1-alpha/2, dofSSR, dofSSE)
    pValue = 1-f.cdf(Fo, dofSSR, dofSSE)
    fout.write('\n')
    fout.write('ANOVA Table for Overall Significance of Fit\n')
    fout.write('Source of Var.\tSum of Squares\tDOF\tMean Square\t\
               F_o\tF_crit\tp\n')
    fout.write('Regression    \t%0.4E\t%g\t%0.4E\t%g\t%g\t%1.4f\n'
               % (SSR, dofSSR, MSR, Fo, Fcrit, pValue))
    fout.write('Error/Resid.  \t%0.4E\t%g\t%0.4E\n' % (SSE, dofSSE, MSE))
    fout.write('Total         \t%0.4E\t%g\n' % (SST, dofSST))

    rsquared = 1 - SSE/SST
    fout.write("\nR Squared = %1.5f\n\n" % (rsquared))

    Sxx = []
    for i in range(0, n):
        Sxx.append(sum((xArray[i, :] - np.mean(xArray[i, :]))**2))

    t0 = [100.1]
    for i in range(1, len(B)):
        t0.append(B[i]/math.sqrt(SigmaSquared*invXTX[i, i]))

    # reject null hypothesis if |tValue| > t(alpha/2,n-1)
    # Equations from page 310, Chapter 13.3 in (old) book
    tStatistic = stats.distributions.t.ppf(alpha/2, n - 1)
    Bsig = []
    for i in range(1, len(B)):
        if abs(t0[i]) > abs(tStatistic):
            Bsig.append(1)
            fout.write("B" + repr(i) + " is significant\n")
        else:
            Bsig.append(0)
            fout.write("B" + repr(i) + " is not significant\n")

    # Confidence intervals for Beta values
    fout.write("Confidence Interval for Beta Values\n")
    for i, B1, C in zip(range(len(B)), B, np.diagonal(parameterCOV)):
        lowB = B1 - stats.t.ppf(1-alpha/2, dofSSE) * math.sqrt(C)
        highB = B1 + stats.t.ppf(1-alpha/2, dofSSE) * math.sqrt(C)
        fout.write("B%i = %g\n" % (i, B1))
        fout.write("%g < B%i < %g with 95%% confidence \n"
                   % (lowB, i, highB))

    # Studentized residuals
    ri = []  # Prepare studentized residuals array
    ei = yArray - yhat
    h = dot(dot(xArray, inv(dot(xArray.transpose(), xArray))),
            xArray.transpose())
    sigmaHatSquared = SSE/(n - len(B))

    for i in range(0, n):
        StudentResidualTemp = ei[i]/(math.sqrt(sigmaHatSquared * (1-h[i, i])))
        # Studentizing equation, from class notes
        ri.append(StudentResidualTemp)

    outliers = 0
    for i in range(0, len(ri)):
        if abs(ri[i]) > 3:
            fout.write("Observation %i is an outlier with \
                Studentized Residual = %g\n" % (i, ri[i]))
            outliers += 1

    if outliers == 0:
        fout.write("No outliers observed in data set\n")

    # Prediction Error Sum of Squares and Studentized Residuals
    PRESS = []
    Rsquared = []
    for i in range(0, n):
        xi = xArray
        xi = np.delete(xi, i, 0)
        yi = yArray
        yi = np.delete(yi, i, 0)
        yihat = dot(xi, B)

        e = []
        for j in range(0, n - 1):
            e.append(yi[j] - yihat[j])

        e = np.array([e]).transpose()
        PRESSTemp = dot(e.transpose(), e)
        PRESS.append(PRESSTemp[0, 0])
        Rsquared.append(1 - PRESS[i] / SST)

    # Start a figure and axes
    fig1, ax1 = plt.subplots(nrows=1, ncols=1)

    # Calculate quantiles and least-square-fit curve
    (quantiles1, values1), (slope1, intercept1, r1) = \
        stats.probplot(ri, dist='norm')

    # plot results
    plt.plot(values1, quantiles1, 'ob', label='Observed Nerve Data')
    plt.plot(quantiles1 * slope1 + intercept1, quantiles1, 'r')
    # define ticks
    ticks_perc = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]

    # transfrom them from precentile to cumulative density
    ticks_quan = [stats.norm.ppf(i/100.) for i in ticks_perc]
    # assign new ticks
    plt.yticks(ticks_quan, ticks_perc)
    # show plot
    plt.grid()
    plt.ylabel('Normal % Probability')
    plt.xlabel('Normalized Residual')
    plt.title('Probability Plot', fontsize=14, fontweight='bold')
    plt.savefig('Residuals-' + DescriptionString + '.png')
    plt.show()
    plt.close()

    return rsquared


def read_file(filename):
    """Read n column contents of file as floats.
    """
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
            xs.append(read_float(1, line))
            ys.append(read_float(0, line))
    return (xs, ys)


def read_float(index, to_read):
    if index is None:
        return None
    else:
        try:
            return float(to_read[index])
        except ValueError:
            print('Float conversion error')


def main():
    """Primary function for reading data and performing analysis."""

    import numpy as np
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--filename')
    parser.add_argument('--kind')
    args = parser.parse_args()
    kind = args.kind

    try:
        filename = args.filename
    except IndexError:
        print('Please provide a file name.')
        return

    (x_list, y_list) = read_file(filename)

    if kind == 'linear':
        x_array = np.transpose(np.array(x_list, ndmin=2))
    elif kind == 'quadratic':
        x2 = [x*x for x in x_list]
        x0 = list(np.ones(len(x_list)))
        x_array = np.transpose(np.array([x0, x_list, x2]))
    else:
        print("Please choose an implemented regression method")
        return

    doRegression(x_array, y_list, "Straw")


if __name__ == "__main__":
    main()
