TO RUN THE SCRIPT:

1. Download all the files in this folder, including the Python script and data files.

2. Make sure all files are in the same directory.

3. Run the Python script (.py) using any Python environment.

4. Install the required libraries if needed (they are listed in the script, e.g. pandas, numpy, matplotlib, scikit-learn, openpyxl).

No additional setup is required.



TO TEST A NEW SUBJECT ON THE PCA PROJECTION:

To visually represent a new patient in the PCA plot (as a yellow data point), follow these steps:

1. Open the file Data_Clustering.xlsx.

2. Append a new row at the end of the table, entering the value 3 in both of the first two columns. This marks the subject as a "test subject" and will display it in yellow on the PCA plot.

3. Then, open the file Dataset_Trial.xlsx.

4. Copy the rest of the patient’s data (all values from the new row in Data_Clustering.xlsx, excluding the first two columns) and append it as a new row at the end of Dataset_Trial.xlsx.

Each row in Dataset_Trial.xlsx corresponds to the ID of the patient as referenced in the main paper.

If you want the new subject to be considered as part of the dataset instead of being tested (e.g., for training or visual comparison purposes), simply replace the 3 values in the first two columns of Data_Clustering.xlsx with:

0 for "No Risk"

1 for "Risk"

2 for "Ambiguous"

This same process can also be applied to the files in the Experiment subfolder if you want to use the cluster-specific scripts and representations described in the paper.



