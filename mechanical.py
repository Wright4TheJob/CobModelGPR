#! /bin/env/python3
# David Wright
# Copyright 2018
# Written for Python 3.7.1
# Weights for each term of objective function: compression, tension, E cost,
# K Specific Heat
# WeightFunction = [0,0,1]
"""Classes for storage and evaluation of mechanical characterization of cob."""
# Properties:
# strengthC: compressive strength, MPa
# strengthT: tensile strength, MPa
# E: elastic modulus, MPa
# cost: typical purchase price, $/yard^3
# K: thermal conductivity, W/(m*K)
# specificHeat: heat capacity, J/K
# density: kg/m^3


def normalize_ratios(ratios):
    """Normalize list so sum of elements equals 1.

    :param ratios: List of elements to normalize
    :type ratios: list of floats or ints
    :returns: list -- Normalized list of floats

    """
    total = sum(ratios)
    if total != 0:
        new_list = [x/total for x in ratios]
    else:
        raise ValueError('Cannot normalize zero-valued list')

    return new_list


def make_plotting_func(function, ratios):
    """Return a lambda function ready for plotting in ternary."""
    return lambda inputs: function(ratios)/100


def clay_saturation(ratios):
    """Determine if clay has filled all voids between sand grains.

    :param inputs: A list of the 3 ingredients used to make the specimen of
    the form [Sand, Clay, Straw]
    :type inputs: list
    :returns: bool -- False if specimen is undersaturated with clay,
    true otherwise.

    """
    ratios = normalize_ratios(ratios)
    clay_frac = ratios[1]
    saturated = clay_frac > 0.1368  # 0.1427
    return saturated


class Cob:
    """Properties of a given batch of cob."""

    def __init__(self):
        """Establish constants for material property calculation."""
        self.sand = {
            "strength_c": 3,
            "strength_t": 0,
            "modulus": 12000,
            "cost": 32.0,
            "thermal_cond": 1,
            "specific_heat": 4,
            "density": 1600}

        self.clay = {
            "strength_c": 1.5,
            "strength_t": 0.5,
            "modulus": 12000,
            "cost": 32.0,
            "thermal_cond": 1,
            "specific_heat": 4,
            "density": 1600}

        self.straw = {
            "strength_c": 0,
            "strength_t": 1.0,
            "modulus": 12000,
            "cost": 32.0,
            "thermal_cond": 1,
            "specific_heat": 4,
            "density": 1600}
        self.predict_function = self.compressive_strength
        self.fiber_effiency = 0.75

    def flexure_strength_matrix(self, ratios):
        """Calculate flexure strength of sand-clay matrix.

        :param inputs: A list of the 3 ingredients used to make the specimen of
        the form [Sand, Clay, Straw]
        :type inputs: list

        :returns:  float -- the maximum force sustained by the specimen
        before failure.

        """
        clay_tensile = self.clay.get('strength_t')
        sand_frac = ratios[0]
        clay_frac = ratios[1]
        strength = -1
        # scale to get ratio of sand and clay alone
        total_vol = sand_frac + clay_frac
        if total_vol == 0:
            # print('No clay or sand included in mix')
            strength = 0
        else:
            clay_frac = clay_frac/total_vol
            sand_frac = sand_frac/total_vol
            strength = 2*clay_tensile*clay_frac + 0.5
        # Extrema:
        # No clay: no flexure strength
        # All clay: high flexure strength, if it doesn't shrinkage crack
        # Below saturation point: crumbly, low strength
        # Above saturation point: more coheisive
        # Bi-linear fit?

        return strength

    def flexure_strength(self, ratios):
        """Calculate flexure (bending) strength for a specimen.

        Uses :func:`CobModel.Mechanical.flexure_strength_matrix` to calculate
        peak flexural load capable of being sustained.

        :param inputs: A list of the 3 ingredients used to make the specimen of
        the form [Sand, Clay, Straw]
        :type inputs: list
        :returns:  float -- the maximum force sustained by the specimen before
        failure.

        """
        straw_tensile = self.straw.get('strength_t')

        ratios = normalize_ratios(ratios)

        straw_frac = ratios[2]
        matrix_flexure = self.flexure_strength_matrix(ratios)
        # linear straw-strength relationship
        slope = 1 * straw_tensile
        strength = matrix_flexure + slope * straw_frac + 0.2
        return strength

    def modulus(self, ratios):
        """Calculate modulus of elasticity for cob.

        :param inputs: A list of the 3 ingredients used to make the specimen of
        the form [Sand, Clay, Straw]
        :type inputs: list
        :returns:  float -- the maximum force sustained by the specimen before
        failure.

        """
        sand_frac = ratios[0]
        clay_frac = ratios[1]
        straw_frac = ratios[2]
        sand_e = self.sand.get('modulus')
        clay_e = self.clay.get('modulus')
        straw_e = self.straw.get('modulus')
        fiber_effiency = 0.98
        # agregate composite model
        matrix_e = (
            0.5*(sand_e * sand_frac + clay_e * clay_frac)
            + 0.5*(sand_e * clay_e)/(
                sand_frac * clay_e + clay_frac * sand_e+0.001
                )
            )

        # Does straw change stiffness? Too much straw would reduce stiffness

        # Fiber composite model: Daniel, Ishai, Pg. 48
        eta = (straw_e - matrix_e)/(straw_e + fiber_effiency*matrix_e)
        modulus = matrix_e * (1 + fiber_effiency*eta*straw_frac)/(
            1-eta*straw_frac)
        # * strawE * strawFrac + matrixModulus * (clayFrac + sandFrac)
        return modulus

    def compressive_strength_matrix(self, ratios):
        """Calculate compressive strength of clay-sand mix in MPa.

        :param inputs: A list of the 3 ingredients used to make the specimen of
        the form [Sand, Clay, Straw]
        :type inputs: list

        :returns:  float -- the compressive strenght of the matrix in MPa.

        """

        clay_frac = ratios[1]
        # Bi-linear model fit
        saturated = clay_saturation(ratios)
        slope = 0
        intercept = 0

        # Independant initial model
        if saturated:
            slope = -0.4629
            intercept = 2.3074
        else:
            slope = 18.14
            intercept = -0.2368

        """
        # Dependant model
        if saturated:
            slope = 0.6877
            intercept = 3.1974
        else:
            slope = 16.5285
            intercept = 0.9363
        """

        compressive_strength = slope * clay_frac + intercept
        # Truncate negative strengths to 0
        if compressive_strength < 0:
            compressive_strength = 0
        return compressive_strength

    def compressive_strength(self, ratios):
        """Calculate the compressive strength of the cob specimen.

        Uses :func:`CobModel.Mechanical.compressive_strength_matrix` to
        calculate peak flexural load capable of being sustained.

        :param inputs: A list of the 3 ingredients used to make the specimen of
        the form [Sand, Clay, Straw]
        :type inputs: list

        :returns:  float -- the compressive strength of the matrix in MPa.

        """
        clay = ratios[1]
        straw_frac = ratios[2]
        # sandC= sand.get('strengthC')
        # clayC = clay.get('strengthC')
        # strawC = straw.get('strengthC')
        matrix_c = self.compressive_strength_matrix(ratios)

        # Independant initial model
        am = 0
        ab = -5889.55
        bm = 0
        bb = 110.46
        c = -0.252

        """
        # Independant Revised Model
        am = 0
        ab = -4344.44
        bm = 0
        bb = 68.89
        c = 0.362
        """
        """
        # Dependant Model
        am = -21364.15
        ab = -2189.97
        bm = -83.962
        bb = 78.8179957608824
        c = -1.22958785322197
        """

        compressive = (am*clay + ab) * straw_frac**2 + \
            (bm*clay + bb) * straw_frac + \
            c + matrix_c
        # (sandFrac + clayFrac) * matrixC + strawFrac * strawC
        return compressive

    def thermal_conductivity(self, ratios):
        """Calculate thermal conductivity of cob."""
        clay_frac = ratios[1]
        straw_frac = ratios[2]
        sand_k = self.sand.get('K')
        clay_k = self.clay.get('K')
        straw_k = self.straw.get('K')
        k_matrix = clay_k * (
            2 * clay_k + sand_k - 2 * (clay_k - sand_k) *
            clay_frac) / (2 * clay_k + sand_k + (clay_k - sand_k) * clay_frac)
        conductivty = k_matrix * (
            2*k_matrix + straw_k - 2*(k_matrix - straw_k) * straw_frac
            )/(2*k_matrix + straw_k + (k_matrix - straw_k) * straw_frac)
        return conductivty

    def cost(self, ratios):
        """Calculate cost of cob."""
        sand_frac = ratios[0]
        clay_frac = ratios[1]
        straw_frac = ratios[2]
        sand_cost = self.sand.get('cost')
        clay_cost = self.clay.get('cost')
        straw_cost = self.straw.get('cost')
        cost = (
            sand_frac * sand_cost +
            clay_frac * clay_cost +
            straw_frac * straw_cost
            )
        return cost

    def shrinkage_cracking_derating(self, ratios):
        """Perform a de-rating of strength based on cob recipe.

        :param inputs: A list of the 3 ingredients used to make the specimen of
        the form [Sand, Clay, Straw]
        :type inputs: list

        :returns: float -- A value from 0 to 1 to be multiplied by strength

        """
        clay_modulus = self.clay.get('modulus')
        straw_frac = ratios[2]
        derating = 1-0.5*(straw_frac)**2 + 0.1 * clay_modulus

        return derating

    def predict(self, mixes, return_std=False):
        """Return a list of values from specified function using mixes.

        Matches syntax useage for scikitlearn GPR functions for
        interchangability. Passing an instance of cob to plotting functions
        and calling cob.predict will return a set of strengths, based on cob
        initialization.
        """
        if len(mixes) == 1:
            return_vals = self.predict_function(mixes[0])
        else:
            return_vals = [self.predict_function(mix) for mix in list(mixes)]
        if return_std:
            return_vals = (return_vals, [0]*len(return_vals))
        return return_vals


def main():
    """Return summary of cob mix."""
    mix = [0.1, 0.3, 0.6]
    this_cob = Cob()
    print(this_cob.cost(mix))


if __name__ == "__main__":
    main()
