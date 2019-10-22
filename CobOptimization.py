#from PyQt5.QtCore import QThread
#import PyQt5.QtCore as QtCore
#from sympy import *
import sys

import sympy as sy
import numpy as np
import math
from numpy.linalg import inv,pinv

def timer(fnc,arg):
	t0 = time.time()
	fnc(arg)
	t1 = time.time()
	return t1-t0

def objectiveFunction(inputs):
	# inputs: [sandFrac,clayFrac,strawFrac,beta]
	beta = inputs[3]
	return -beta

def sandMax(inputs):
	# inputs: [sandFrac,clayFrac,strawFrac,beta]
	# Less than 1
	sandFrac = inputs[0]
	return sandFrac - 1.

def sandMin(inputs):
	# inputs: [sandFrac,clayFrac,strawFrac,beta]
	# Greater than 0
	sandFrac = inputs[0]
	return -sandFrac

def clayMax(inputs):
	# inputs: [sandFrac,clayFrac,strawFrac,beta]
	# Less than 1
	clayFrac = inputs[1]
	return clayFrac - 1.

def clayMin(inputs):
	# inputs: [sandFrac,clayFrac,strawFrac,beta]
	# Greater than 0
	clayFrac = inputs[1]
	return -clayFrac

def strawMax(inputs):
	# inputs: [sandFrac,clayFrac,strawFrac,beta]
	# Less than 1
	strawFrac = inputs[2]
	return strawFrac - 1.

def strawMin(inputs):
	# inputs: [sandFrac,clayFrac,strawFrac,beta]
	# Greater than 0
	strawFrac = inputs[2]
	return -strawFrac

def totalVolume(inputs):
	return (sum(inputs) - 1.)**2

def tensileInequality(inputs):
	tensileMax = 100 # TODO: Get model to find max theoretical of each property
	beta = inputs[3]
	result = beta - model.tensileStrength(inputs)/tensileMax
	return result

def main():
    import NumericalOptimization as opt
    import CobModel as model

    # Inputs: lamda function of objective function to minimize
    # 		list of design parameters. Design parameters are only input to objective and constraint functions
    # optional:
    # 		list of pairs (lambda function of inequality constraint, max value)
    # 		list of pairs (lambda function of equality constraint, target)
    #		max iterations, damping, type of algorithm
    # output:
    # Recipe: (Sand, Clay, Straw)
    initialRecipe = [0.5,0.3,0.2]
    # Design parameter list creation
    initialPosition = initialRecipe
    initialBeta = -100.0
    initialPosition.append(initialBeta)
    optimumParameters = list(initialPosition)

    # Constraint functions
    # Reasonableness:
    # No ingredient may be more than 100% of mix
    # Sum of ingredients must equal 100%
    # Manufacturability
    # Straw < 40%
    # Clay > 5%
    #g1 = 'volSand-1' # less than or equal to 0
    #g2 = 'volClay-1' # less than or equal to 0
    #g3 = 'volStraw-1' # less than or equal to 0
    #g4 = '-volSand' # less than or equal to 0
    #g5 = '-volClay' # less than or equal to 0
    #g6 = '-volStraw' # less than or equal to 0
    # normalized material property > b
    #g7 = str(-tensile/tensileMax + b) # Tensile Function
    #g8 = str(-E/EMax + b) # Modulus Function
    #g9 = str(-compressive/compressiveMax + b)
    #g10 = str(-1+conductivity/conductivityMax + b)
    #g11 = str(cost/costMax - 1 + b)
    #h1 = 'volSand + volClay + volStraw - 1' # equal to 0
    inequalityConstraints = [sandMax,sandMin,clayMax,clayMin,strawMax,strawMin,tensileInequality]
    equalityConstraints = [totalVolume]
    # Optimization parameters
    rp = 1
    rpMax = 10000000
    n = 0

    while rp < rpMax:
    	n = n+1
    	print("Running iteration %i"%(n))
    	(goodness,optimumParameters) = opt.constrainedMinimum(
    		objectiveFunction,optimumParameters,
    		inequalityConstraints = inequalityConstraints,
    		equalityConstraints = equalityConstraints,
    		rp=rp,
    		echo=False,
    		damping=.001,
    		epsilon=0.0001,
    		nMax=100000,
    		printResults=True,
    		method='ExteriorPenalty')# ExteriorPenalty, InteriorLinearExtended, InteriorInverseBarrier
    	rp = rp*1.5

    print("----Results----")
    print("Recipe:")
    print("Sand: %2.2f"%(optimumParameters[0]*100.))
    print("Clay: %2.2f"%(optimumParameters[1]*100.))
    print("Straw: %2.2f"%(optimumParameters[2]*100.))
    print("Beta: %2.2f"%(optimumParameters[3]))
    print("---Properties---")
    print("Tensile Strength: %2.4f"%(model.tensileStrength(optimumParameters)))


if __name__ == "__main__":
    main()
