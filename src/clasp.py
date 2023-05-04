def ClaSP(time_series, tcps):
    
    plt.plot(time_series)
    # plt.show()
    clasp = BinaryClaSPSegmentation()
    regimes = clasp.fit_predict(time_series)
    clasp.plot(gt_cps=tcps, heading="Segmentation", ts_name="ACC", file_path="ts_segmentation.png")
    
    return regimes