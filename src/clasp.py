from claspy.segmentation import BinaryClaSPSegmentation
from claspy.data_loader import load_tssb_dataset

def ClaSP(time_series, tcps):
    
    plt.plot(time_series)
    # plt.show()
    clasp = BinaryClaSPSegmentation()
    regimes = clasp.fit_predict(time_series)
    clasp.plot(gt_cps=tcps, heading="Segmentation", ts_name="ACC", file_path="ts_segmentation.png")
    
    return regimes

dataset, window_size, true_cps, time_series = load_tssb_dataset(names=("CricketX", )).iloc[0, :]
# clusters = ClaSP(time_series, true_cps)
# print(clusters)