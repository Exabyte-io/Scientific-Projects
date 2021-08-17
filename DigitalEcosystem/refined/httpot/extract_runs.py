import os
import shutil

image_gather_dir = "images"
data_gather_dir = "collated_results"
os.makedirs(image_gather_dir, exist_ok=True)

plot_filename = "plot.jpeg"
for dirpath, dirnames, filenames in os.walk("."):
    if plot_filename in filenames:
        splitpath = dirpath.split(os.sep)
        rowfilter, colfilter, fingerprint, model, k_choice = splitpath[1], splitpath[2], splitpath[3], splitpath[4], \
                                                             splitpath[5]
        srcname = os.path.join(dirpath, plot_filename)
        destname = os.path.join(image_gather_dir, "-".join([rowfilter, colfilter, fingerprint, model, k_choice]) + ".jpeg")
        print(srcname, "->", destname)
        shutil.copy2(srcname, destname)

    if 'result.csv' in filenames:
        for file in ['log.out', 'optimized_model.py', 'predictions.csv', 'result.csv', 'run_job.py', 'stderr.out', 'job.pbs']:
            srcname = os.path.join(dirpath, file)
            destdir = os.path.join(data_gather_dir, *dirpath.split(os.sep)[1:])
            os.makedirs(destdir, exist_ok=True)

            destname = os.path.join(destdir, file)
            try:
                shutil.copy2(srcname, destname)
                print(srcname, "->", destname)
            except FileNotFoundError:
                pass


