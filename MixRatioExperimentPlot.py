# https://github.com/marcharper/python-ternary


def valid_mixes(mixes):
    for mix in mixes:
        if not valid_mix(mix):
            print('Mix does not sum to 1: ')
            print(mix)
            print('Sum is %2.4f' % (sum(mix)))


def soil2clay(mix, clay_frac):
    sand = mix[0]
    straw = mix[2]
    soil = mix[1]
    total_sand = sand + soil*(1-clay_frac)
    clay = soil*clay_frac
    return (total_sand, clay, straw)


def valid_mix(mix, epsilon=0.05):
    valid = True
    error = abs(100-sum(mix))
    if error > epsilon:
        valid = False
    return valid


def plot_mixes(current_mix, mixes, binder='Clay'):
    import ternary

    #  Scatter Plot
    scale = 100
    figure, tax = ternary.figure(scale=scale)
    tax.get_axes()
    # tax.set_title("Scatter Plot", fontsize=20)
    tax.boundary(linewidth=2.0)
    tax.gridlines(multiple=10, color="blue")

    # Plot a few different styles with a legend
    tax.scatter(current_mix, marker='o', color='k', label="Standard Mix")
    tax.scatter(mixes, marker='+', color='r', label="Test Mixes")
    tax.legend()
    tax.ticks(axis='lbr', linewidth=1, multiple=5)
    fontsize = 18
    tax.bottom_axis_label("Sand", fontsize=fontsize)
    tax.right_axis_label(binder, fontsize=fontsize)
    tax.left_axis_label("Straw", fontsize=fontsize)
    # plt.ylim(ymin=-20)
    # Remove default Matplotlib Axes
    # ax.tight_layout()
    tax.clear_matplotlib_ticks()
    tax.savefig("MixRatios"+binder+".png", dpi=300)


def main():
    # point = (bottomAxis,rightAxis,leftAxis)
    soil_clay_frac = 0.3
    # (Sand, Soil, Straw)
    current_mix_soil = (65.25, 33, 0.75)
    current_mix_clay = soil2clay(current_mix_soil, soil_clay_frac)
    clay_mixes = [
        (81.01, 18.65, 0.35),
        (44.75, 54.88, 0.37),
        (26.27, 73.30, 0.42),
        (0.00, 99.58, 0.42)]
    straw_mixes = [
        (70.32, 28.52, 1.17),
        (68.46, 30.70, 0.84),
        (70.69, 28.92, 0.39),
        (68.55, 31.45, 0.00)]
    mixes_soil = clay_mixes + straw_mixes
    valid_mixes(mixes_soil)
    mixes_clay = [soil2clay(mix, soil_clay_frac) for mix in mixes_soil]
    valid_mixes(mixes_clay)

    plot_mixes([current_mix_soil], mixes_soil, binder="Soil")
    plot_mixes([current_mix_clay], mixes_clay, binder="Clay")


if __name__ == "__main__":
    main()
