import sympy as sy
import numpy as np
import random
from sympy import *
import scipy.optimize
import scipy
import math
#from sympy.mpmath import *

def vlen(inputs):
    totalSquared = 0
    for input in inputs:
        totalSquared += input*input
    result = math.sqrt(totalSquared)
    return result

def degToRad(deg):
    rad = deg/360.0
    return rad

def make2dList(rows, cols):
    a=[]
    for row in xrange(rows): a += [[0]*cols]
    return a

def variableSymbols(variables):
    if variables:
        variableSymbols = []
        if isinstance(variables[0],str) == True:
            for variable in variables:
                variableSymbols.append(symbols(variable))
        else:
            variableSymbols = variables

    return variableSymbols

def expressionSymbols(expression):

    if isinstance(expression, str):
        expression = sy.sympify(expression)
    return expression

def gradient(function,inputs,delta=0.0001,normalize=False):
    '''Returns a list of partial gradients of the function around the input point.'''
    # Inputs: function is a python function that accepts only a list of inputs as arguments
    # Inputs is a list representing the point at which to evaluate the function.
    # Optional: delta is the numerical step size of the gradient approximation
    # Normalize returns the slope of each partial of the gradient divided by the total slope

    slopeValues = []
    for i in range(0,len(inputs)):
        negativeInputs = list(inputs)
        negativeInputs[i] = float(negativeInputs[i]) - float(delta)
        negativePoint = function(negativeInputs)

        positiveInputs = list(inputs)
        positiveInputs[i] = float(positiveInputs[i]) + float(delta)
        positivePoint = function(positiveInputs)

        slope = (positivePoint - negativePoint)/(2*delta)
        slopeValues.append(slope)

    if normalize == True:
        totalSlope = vlen(slopeValues)
        for i in range(0,len(slopeValues)):
            slopeValues[i] = slopeValues[i]/totalSlope
    return slopeValues

def steepestDescentMinimum(function,startingPoint,epsilon=0.0001,nMax=100,damping=1,echo=False,parabolaFitStepSize = 0.1,constantStepSize = 0.1,**kwargs):
    '''Minimizes output of function using steepest descent method.'''
    # Inputs: python function which returns a single value and takes an input of a list of values
    # Variables is a list of text inputs for each input variable
    # StartingPoint is a vector of intial points for each input variable
    # Convergence and timeout parameters are optional
    alpha = [-parabolaFitStepSize,0,parabolaFitStepSize]
    i = 0
    import FunctionApproximation as approx

    # Loop
    shouldContinue = True
    position = startingPoint
    objectiveValue = function(position)
    # print("starting loop...")
    # Print current iteration results
    if echo == True:
        headerString = "Iteration\tPosition\t"
        headerString += "Gradient\t"
        headerString += "F(x)"
        print(headerString)

    while shouldContinue == True:
        i = i+1
        # Get gradient at position
        # print("About to get gradient")
        slopeList = gradient(function,position)
        # print("fitting polynomial...")
        # Get three points in that direction at positions of alpha
        functionValues = []
        for alphaValue in alpha:
            testLocation = []
            for oldPosition, slope in zip(position,slopeList):
                testLocation.append(oldPosition-slope*alphaValue)
            functionValues.append(function(testLocation))
        # Fit parabola to curve
        C = approx.threePointQuadraticApprox(alpha, functionValues)
        # Check parabola is concave up
        # Calculate alpha that gives minimum
        alphaStar = 0.0
        if C[2] < 0:
            print("Fitted parabola is concave down. Minimum alpha value is not bounded.")
            alphaStar = constantStepSize
        elif abs(C[2]) < 0.001:
            print("Shallow gradient, using constant step size")
            alphaStar = constantStepSize
        else:
            (alphaStar,bestY) = minimizeParabola(C)
        # Move to position of calculated alpha
        newPosition = []
        for oldPosition, slope in zip(position,slopeList):
            newPosition.append(oldPosition-slope*damping*alphaStar)
        lastPosition = position
        position = newPosition
        objectiveValueLast = objectiveValue
        objectiveValue = function(position)

        # Print current iteration results
        if echo == True:
            resultsString = "%i        \t" %(i)
            resultsString += "{}\t".format(position)
            resultsString += "{}\t".format(slopeList)
            resultsString += "%2.6f" % (objectiveValue)
            print(resultsString)

        # Check convergence
        deltaObjective = objectiveValueLast - objectiveValue
        #print("Delta Objective = %2.4f" % (float(deltaObjective)))
        if abs(deltaObjective) <= epsilon:
            shouldContinue = False
            print("Local Optimium found")

        #print("About to check iteration maximum")
        if i > nMax:
            print("Function timed out. Returning final result")
            shouldContinue = False

    print("#### - Results - ####")
    print("Position is:")
    print(position)
    print("F = %2.6f" % (objectiveValue))
    return (objectiveValue, position)

def evaluateExteriorPenalty(function, position,
    inequalityConstraints=[],
    equalityConstraints=[], rp=1):
    '''Returns a float at the location selected with constraint penalties.'''

    objectiveValue = function(position)

    penalty_total = 0
    for constraint in inequalityConstraints:
        penalty = constraint(position)
        penalty = max(0,penalty)**2
        penalty_total +=  penalty

    for constraint in equalityConstraints:
        penalty = constraint(position)**2
        penalty_total +=  penalty

    totalValue = objectiveValue + rp * penalty_total
    result = totalValue

    return result

def evaluateLinearExtendedPenalty(
    function, position,
    inequalityConstraints=[],
    equalityConstraints=[],
    rp=1.0,
    epsilon = -9999):
    """returns a float at the location selected with constraint penalties"""

    if epsilon == -9999:
        epsilon = -0.2*np.sqrt(1/rp)

    rpPrime = 1/rp
    objectiveValue = function(position)

    inconstraintValue = 0
    for constraint in inequalityConstraints:
        newConstraintValue = constraint(position)
        if newConstraintValue > epsilon:
            inconstraintValue += - (2*epsilon - newConstraintValue)/epsilon**2
        else:
            inconstraintValue = inconstraintValue - 1/newConstraintValue

    constraintValue = 0
    for constraint in equalityConstraints:
        newConstraintValue = constraint(position)**2
        constraintValue = constraintValue + newConstraintValue

    totalValue = objectiveValue + inconstraintValue/rp + constraintValue*rp
    result = totalValue

    return result

def evaluateInteriorInverseBarrier(
    function, position,
    inequalityConstraints=[],
    equalityConstraints=[],
    rp=1.0):
    """returns a float at the location selected with constraint penalties"""

    objectiveValue = function(position)

    ineq_constraint_penalty = 0
    for constraint in inequalityConstraints:
        constraint_value = constraint(position)
        if ineq_constraint_penalty <= 0:
            ineq_constraint_penalty += - 1/constraint_value
        else:
            ineq_constraint_penalty += 100*rp * constraint_value

    eq_constraint_penalty = 0
    for constraint in equalityConstraints:
        constraint_value = constraint(position)**2
        eq_constraint_penalty += constraint_value

    result = objectiveValue + ineq_constraint_penalty/rp + rp * eq_constraint_penalty

    return result

def peanalizerForMethod(evaluator):
    def peanalizer(objective, inequalityConstraints, equalityConstraints):
        return lambda position: evaluator(objective, inequalityConstraints=inequalityConstraints, equalityConstraints=equalityConstraints, position=position)

    return peanalizer

#PenalizeExterior = peanalizerForMethod(evaluateExteriorPenalty)
#PenalizeInterior = peanalizerForMethod(evaluateInteriorInverseBarrier)

#minimizeUsing(PenalizeExterior(someObjective, [blah blah], [blah]))

def constrainedMinimum(function,startingPoint,
    inequalityConstraints=[],
    equalityConstraints=[],
    rp=1,
    method='ExteriorPenalty',
    echo=False,
    damping=0.1,
    epsilon=0.0001,
    nMax=100,
    parabolaFitStepSize = 0.1,
    constantStepSize = 0.1,
    printResults=True,
    **kwargs):
    '''minimizes the given function for n variables subject to boundary constraints'''
    # Input: function whose only argument is a list of values and which returns a single value
    # StartingPoint is a list of values corrosponding to the number of variables
    # Constraints are functions which take a list of values and return a single value
    # Inequality constraints return less than 0 when valid, equality equal 0
    # Method options: 'ExteriorPenalty', 'InteriorPenalty', 'InteriorInverseBarrier','InverseLog', 'InteriorLinearExtended', 'QuadraticExtended'
    if method == 'ExteriorPenalty':
        penalizedFunction = lambda position: evaluateExteriorPenalty(function,
            inequalityConstraints=inequalityConstraints,
            equalityConstraints=equalityConstraints,
            position = position,
            rp = rp)
    elif method == 'InteriorLinearExtended':
        penaltyEvaluator = evaluateLinearExtendedPenalty
        penalizedFunction = lambda position: evaluateLinearExtendedPenalty(function,
            inequalityConstraints=inequalityConstraints,
            equalityConstraints=equalityConstraints,
            position = position,
            rp = rp,
            epsilon = -9999)
    elif method == 'InteriorInverseBarrier':
        penaltyEvaluator = evaluateInteriorInverseBarrier
        penalizedFunction = lambda position: evaluateInteriorInverseBarrier(function,
            inequalityConstraints=inequalityConstraints,
            equalityConstraints=equalityConstraints,
            position = position,
            rp = rp)
        #penalizedFunction = lambda position: penaltyEvaluator(function, ...)
    else:
        print('The method ' + method + ' is not implemented yet.')
        return

    (optimum, position) = steepestDescentMinimum(penalizedFunction, startingPoint,
    epsilon=epsilon,
    nMax=nMax,
    damping=damping,
    echo=echo,
    parabolaFitStepSize = parabolaFitStepSize,
    constantStepSize = constantStepSize,**kwargs)

    return (optimum, position)

def minimizeCubic(c):
    import FunctionApproximation as approx

    # Inputs: Coefficients for polynomial equation according to the form C0 + C1*x + C2*x^2 + C3*x^3
    # Outputs: Values of x and y where y is minimized
    a = 3*c[3]
    b = 2*c[2]
    d = c[1]
    insideSqareroot = np.float64(b*b-4*a*d)
    if insideSqareroot < 0:
        print("Minimize Cubic function encountered imaginary square root. Aborting.")
        return
    x1 = (-b+np.sqrt(insideSqareroot))/(2*a)
    x2 = (-b-np.sqrt(insideSqareroot))/(2*a)

    x = 0
    y = 0

    y1 = approx.getValueOfPoly(c,x1)
    y2 = approx.getValueOfPoly(c,x2)
    if y1 < y2:
        x = x1
        y = y1
    elif y1 > y2:
        x = x2
        y = y1
    else:
        x = x1
        y = y1
        print("More than one solution in Minimize Cubic")

    return (x,y)

def minimizeParabola(c):
    import FunctionApproximation as approx

    # Inputs: Coefficients for polynomial equation according to the form C0 + C1*x + C2*x^2...
    # Outputs: Values of x and y where y is minimized
    minX = -c[1]/(2*c[2])

    minY = approx.getValueOfPoly(c,minX)
    return (minX,minY)

def convertToPenaltyFunction(coreFunction,constraints,R=1):
    constraintsToSum = []
    newObjective = coreFunction + " - %2.4f*(" % (R)
    for i in range(0,len(constraints)):
        constraint = constraints[i]
        if i == 0:
            newObjective = newObjective + "1/(" + constraint + ")"
        else:
            newObjective = newObjective + " + 1/(" + constraint + ")"


    newObjective = newObjective + ")"
    return newObjective
