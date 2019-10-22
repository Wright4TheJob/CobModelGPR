import numpy as np
import csv
from numpy.linalg import pinv
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d

from regression import doRegression
from regression import twofactorANOVA
from regression import plotConfidence

__author__ = 'davidwright'

"""
General program for regression analysis of cob mix data.
Data file to be read in comma separated, columns:
Sand, Soil, Straw, Strength
Each data point in a separate row
"""

# Data Importing
SoilList = []
StrawList = []
StrengthList = []

print('Loading Data...')

with open('all_data_comp.txt', 'r') as fin:
    next(fin)  # skip headings
    reader = csv.reader(fin, delimiter='\t')
    for soil, straw, strength in reader:
        SoilList.append(float(soil))
        StrawList.append(float(straw))
        StrengthList.append(float(strength))

n = len(StrawList)

fout = open('regression_summary.txt', 'w')

# Program Settings
y = StrengthList

n = len(y)
rSquared = []
TestNames = ['Linear', 'Quadratic']

# Preliminary ANOVA Test
print("Performing ANOVA...")
twofactorANOVA(SoilList, StrawList, y, 'Soil', 'Straw')

# Quadratic Model
print('Fitting quadratic model...')
fout.write("\n Quadratic Additive Model\n")
fout.write(
    "Model: y = B0 + B1*x1 + B2*x2 + B3*x1*x2+ B4*x1^2 + B5*x2^2 + error\n")
ones = np.ones(n)
xArray = np.array([ones, SoilList, StrawList,
                  [a*b for a, b in zip(SoilList, StrawList)],
                  np.square(SoilList), np.square(StrawList)]).transpose()

quality = doRegression(xArray, y, 'Quadratic')
rSquared.append(quality)

XTX = np.dot(xArray.transpose(), xArray)
fout.write("XTX Array: \n")
for i in range(0, len(XTX[:, 0])):
    for j in range(0, len(XTX[0, :])):
        fout.write("%4.3f\t" % XTX[i, j])
    fout.write("\n")

fout.write("R^2 Value: %1.4f\n" % (rSquared[-1]))

# Response Surface Visualization
# model: y = B0 + B1*x1 + B2*x2 + B3*x1*x2+ B4*x1^2 + B5*x2^2
params = 5
x1 = SoilList
x2 = StrawList
x1squared = []
x2squared = []
x1x2 = []
for i in range(0, n):
    x1squared.append((x1[i])**2)
    x2squared.append((x2[i])**2)
    x1x2.append(x1[i]*x2[i])

x = np.zeros((n, params))
for i in range(0, n):
    x[i, 0] = 1
    x[i, 1] = x1[i]
    x[i, 2] = x2[i]
    x[i, 3] = x1squared[i]
    x[i, 4] = x2squared[i]
#     x[i, 5] = x1x2[i]

xtx = np.dot(x.transpose(), x)
xtxinv = pinv(xtx)
xty = np.dot(x.transpose(), y)
B = np.dot(xtxinv, xty)

points = 120
x1Spacing = np.linspace(0, 1.2*max(x1), points)
x2Spacing = np.linspace(0, 1.2*max(x2), points)
x = np.outer(x1Spacing, np.ones(points))
ySpacing = np.outer(x2Spacing, np.ones(points))
ySpacing = ySpacing.copy().T

z = np.zeros((points, points))
# print("Z = " + repr(z))
for i in range(0, len(x1Spacing)):
    for j in range(0, len(x2Spacing)):
        z[i, j] = B[0] +\
            B[1]*x1Spacing[i] + \
            B[2]*x2Spacing[j] + \
            B[3]*(x1Spacing[i])**2 + \
            B[4]*(x2Spacing[j])**2  # + \
#           B[5]*x1Spacing[i]*x2Spacing[j]
Z = z.reshape(x.shape)

fig = plt.figure()
fig.set_size_inches(6, 4)
ax = fig.gca(projection='3d')

surf = ax.plot_wireframe(x, ySpacing, Z, rstride=10, cstride=10)
# surf = ax.plot_surface(x, ySpacing, Z, rstride=5, cstride=5,
#                        cmap=cm.copper_r,
#                       linewidth=0, antialiased=False)
# fig.colorbar(surf, shrink=0.5, aspect=5)

ax.scatter3D(SoilList, StrawList, y)
ax.set_xlabel('Soil Content')
ax.set_ylabel('Straw Content')
ax.set_zlabel('Bending Strength')
plt.savefig('regression_surface.png', dpi=300)
plt.show()
plt.close()

fout.close()
