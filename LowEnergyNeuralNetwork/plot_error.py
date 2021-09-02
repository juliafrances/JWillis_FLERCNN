def plot_1d_binned_slices(truth, error1, error2=None,
                       xarray1=None,xarray2=None,truth2=None,\
                       plot_resolution=False, use_fraction = False,\
                       bins=10,xmin=None,xmax=None,style="contours",\
                       x_name = "Zenith", x_units = "",\
                       error1_name = "Reco 1", error2_name = "Reco 2",\
                       error1_weight = None, error2_weight = None,
                       save=True,savefolder=None):
    """Plots different energy slices vs each other (systematic set arrays)
    Receives:
        truth = 1D array with truth values
        error1 = 1D array that has reconstructed results
        error2 = optional, 1D array that has an alternate reconstructed results
        xarray1 = optional, 1D array that the error1 variable (or resolution) will be plotted against, if none is given, will automatically use truth1
        xarray2 = optional, 1D array that the error2 variable (or resolution2) will be plotted against, if none is given, will automatically use xarray1
        truth2 = 1D array with truth values used to calculate resolution2
        plot_resolution = use resolution (reco - truth) instead of just reconstructed values
        use_fraction = bool, use fractional resolution instead of absolute, where (reco - truth)/truth
        style = "errorbars" is only string that would trigger change (to errorbar version), default is contour plot version
        bins = integer number of data points you want (range/bins = width)
        xmin = minimum truth value to start cut at (default = find min)
        xmax = maximum truth value to end cut at (default = find max)
        x_name = variable for x axis (what is the truth)
        x_units = units for truth/x-axis variable
        error1_name = name for reconstruction 1
        error2_name = name for reconstruction 2
        error1_weight = 1D array for error1 weights, if left None, will not use
        error2_weight = 1D array for error2 weights, if left None, will not use
    Returns:
        Scatter plot with truth bins on x axis (median of bin width)
        y axis has median of resolution or absolute reconstructed value with error bars containing given percentile
    """

    percentile_in_peak = 68.27 #CAN CHANGE
    left_tail_percentile  = (100.-percentile_in_peak)/2
    right_tail_percentile = 100.-left_tail_percentile

    # if no xarray given, automatically use truth
    if xarray1 is None:
        xarray1 = truth
    if xmin is None:
        xmin = min(xarray1)
        print("Setting xmin based on xarray1 (or truth)--not taking into account xarray2")
    if xmax is None:
        xmax = max(xarray1)
        print("Setting xmax based on xarray1 (or truth)--not taking into account xarray2")

    ranges  = numpy.linspace(xmin,xmax, num=bins)
    centers = (ranges[1:] + ranges[:-1])/2.

    # Calculate resolution if plot_resolution flag == True
    if plot_resolution:
        if use_fraction:
            yvariable = ((error1-truth)/truth) # in fraction
        else:
            yvariable = (error1-truth)
    else: #use reco directly, not resolution
        y_variable = error1
        assert use_fraction==False, "Flag for fractional resolution only, not doing resolution here"

    medians  = numpy.zeros(len(centers))
    err_from = numpy.zeros(len(centers))
    err_to   = numpy.zeros(len(centers))

    #Compare to second reconstruction if given
    if error2 is not None:
        #check if some variables exist, if not, set to match error1's
        if truth2 is None:
            truth2 = truth1
        if xarray2 is None:
            xarray2 = xarray1

        if plot_resolution:
            if use_fraction:
                yvariable2 = ((error2-truth2)/truth2)
            else:
                yvariable2 = (error2-truth2)
        else:
            yvariable2 = error2
        medians2  = numpy.zeros(len(centers))
        err_from2 = numpy.zeros(len(centers))
        err_to2   = numpy.zeros(len(centers))

    # Find median and percentile bounds for data
    for i in range(len(ranges)-1):

        # Make a cut based on the truth (binned on truth)
        var_to   = ranges[i+1]
        var_from = ranges[i]
        cut = (xarray1 >= var_from) & (xarray1 < var_to)
        assert sum(cut)>0, "No events in xbin from %s to %s for error1, may need to change xmin, xmax, or number of bins or check truth/xarray1 inputs"%(var_from, var_to)
        if error2 is not None:
            cut2 = (xarray2 >= var_from) & (xarray2 < var_to)
            assert sum(cut2)>0, "No events in xbin from %s to %s for error2, may need to change xmin, xmax, or number of bins or check truth2/xarray2 inputs"%(var_from, var_to)

        #find number of error1 (or resolution) in this bin
        if error1_weight is None:
            lower_lim = numpy.percentile(yvariable[cut], left_tail_percentile)
            upper_lim = numpy.percentile(yvariable[cut], right_tail_percentile)
            median = numpy.percentile(yvariable[cut], 50.)
        else:
            import wquantiles as wq
            lower_lim = wq.quantile(yvariable[cut], error1_weight[cut], left_tail_percentile)
            upper_lim = wq.quantile(yvariable[cut], error1_weight[cut], right_tail_percentile)
            median = wq.median(yvariable[cut], error1_weight[cut])

        medians[i] = median
        err_from[i] = lower_lim
        err_to[i] = upper_lim

        #find number of error2 (or resolution2) in this bin
        if error2 is not None:
            if error2_weight is None:
                lower_lim2 = numpy.percentile(yvariable2[cut2], left_tail_percentile)
                upper_lim2 = numpy.percentile(yvariable2[cut2], right_tail_percentile)
                median2 = numpy.percentile(yvariable2[cut2], 50.)
            else:
                import wquantiles as wq
                lower_lim2 = wq.quantile(yvariable2[cut2], error2_weight[cut2], left_tail_percentile)
                upper_lim2 = wq.quantile(yvariable2[cut2], error2_weight[cut2], right_tail_percentile)
                median2 = wq.median(yvariable2[cut2], error2_weight[cut2])

            medians2[i] = median2
            err_from2[i] = lower_lim2
            err_to2[i] = upper_lim2

    # Make plot
    plt.figure(figsize=(10,7))

    # Median as datapoint
    # Percentile as y error bars
    # Bin size as x error bars
    if style is "errorbars":
        plt.errorbar(centers, medians, yerr=[medians-err_from, err_to-medians], xerr=[ centers-ranges[:-1], ranges[1:]-centers ], capsize=5.0, fmt='o',label="%s"%error1_name)
        #Compare to second reconstruction, if given
        if error2 is not None:
            plt.errorbar(centers, medians2, yerr=[medians2-err_from2, err_to2-medians2], xerr=[ centers-ranges[:-1], ranges[1:]-centers ], capsize=5.0, fmt='o',label="%s"%error2_name)
            plt.legend(loc="upper center")
    # Make contour plot
    # Center solid line is median
    # Shaded region is percentile
    # NOTE: plotted using centers, so 0th and last bins look like they stop short (by 1/2*bin_size)
    else:
        alpha=0.5
        lwid=3
        cmap = plt.get_cmap('Blues')
        colors = cmap(numpy.linspace(0, 1, 2 + 2))[2:]
        color=colors[0]
        cmap = plt.get_cmap('Oranges')
        rcolors = cmap(numpy.linspace(0, 1, 2 + 2))[2:]
        rcolor=rcolors[0]
        ax = plt.gca()
        ax.plot(centers, medians,linestyle='-',label="%s median"%(error1_name), color=color, linewidth=lwid)
        ax.fill_between(centers,medians, err_from,color=color, alpha=alpha)
        ax.fill_between(centers,medians, err_to, color=color, alpha=alpha,label=error1_name + " %i"%percentile_in_peak +'%' )
        if error2 is not None:
            ax.plot(centers,medians2, color=rcolor, linestyle='-', label="%s median"%error2_name, linewidth=lwid)
            ax.fill_between(centers,medians2,err_from1, color=rcolor, alpha=alpha)
            ax.fill_between(centers,medians2,err_to2, color=rcolor,alpha=alpha,label=error2_name + " %i"%percentile_in_peak +'%' )

    # Extra features to have a horizontal 0 line and trim the x axis
    plt.plot([xmin,xmax], [0,0], color='k')
    plt.xlim(xmin,xmax)

    #Make pretty labels
    plt.xlabel("%s %s"%(x_name,x_units))
    if plot_resolution:
        if use_fraction:
            plt.ylabel("Fractional Resolution: \n (reconstruction - truth)/truth")
        else:
            plt.ylabel("Resolution: \n reconstruction - truth %s"%x_units)
    else:
        plt.ylabel("Reconstructed %s %s"(x_name,x_units))

    # Make a pretty title
    title = "%s Dependence for %s Error"%(x_name,error1_name)
    if error2 is not None:
        title += " and %s"(error2_name)
    if plot_resolution:
        title += " Resolution"
    plt.title("%s"%(title))

    # Make a pretty filename
    savename = "%s"%(x_name.replace(" ",""))
    if use_fraction:
        savename += "Frac"
    if plot_resolution:
        savename += "Resolution"
    if error2 is not None:
        savename += "_Compare%s"%(error2_name.replace(" ",""))
    if save == True:
        plt.savefig("%s/%s.png"%(savefolder,savename))
