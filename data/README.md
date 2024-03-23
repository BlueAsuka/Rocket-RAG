The data folder includes all data used in this project. The `instances` and `inference` folders will be generated automatically after running the `data_processing.py` file.

- `raw`: the raw data for this project to start
- `instances`: the instances to enable the system to refer and deriate on. 
- `inference`: the samples for testing the inference capability of the system  

The raw data is original from the following link:

https://cord.cranfield.ac.uk/articles/dataset/Data_set_for_Data-based_Detection_and_Diagnosis_of_Faults_in_Linear_Actuators_/5097649

The original file is converted from .mat format into .csv format using the `notebooks/file_converter.py` and not listed in the raw data folder. It is recommanded to use the .csv file as the beginning of the project (It is time-consuming to convert the data).  

There is a file for illustrating how the raw data is collected in detailed named `Datadescription.pdf` in the same folder. If you have interesting on how the dataset is constructed, feel free to check it out. In short, the name of each file suggest a state of the current linear actuator includeing: **normal**, **spalling**, **backlash** and **lack of lubrication**. The number attached to the file name is the degradation level. For example `spalling1.csv` means the degradation level is 1 at the state of spalling. 

Only data under trapezodial motion profile is selected. The figure of the motion profile is shown in the figure below.
![](/assests/trapezoidal_motion_profile.PNG)

In `instances` and `inference`, there are three subfolders namely `20kg`, `40kg` and `-40kg`, which are the three different loads. The data in each subfolder is collected under the same load but in different states.

According to the data description, for each type of fualt under a specific load, there are 5 (experiement trials) $\times$ 10 (repeated times in each experiment trials) groups of data. Within these 50 files in instances, 10 files will be randomly selected to be the inference set for evaluating the system later. 