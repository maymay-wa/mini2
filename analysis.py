# Mayer Unterberg
# This function will define every finger movement as a block of time that is 200 ms before the starting point of the movement, 1 ms of the starting point, and 1000 ms after the starting point of the movement. 
# 
# Such that each finger movement will extract 1201 indices of brain data that relate to the movement of that finger
# Gather all the signals for each event and average across them such that there is a list of averaged brain signals of length 1201 for each unique finger movement. 
# 
# For example – let’s say the patient was asked to move finger 1 one hundred times (100 trials), there are now 100 samples of a block of time in brain data of length 1201. Average across the 100 trials such that we only end up with 1 list of length 1201 for that finger.
# 
# This function will plot the averaged brain response for each finger
# 
# This function will return a matrix with the averaged brain response list for each finger – matrix size will be 5x1201 – five finger events, and the length of brain data for each event – 1201 – order by finger – finger 1 should be the first row.
import pandas as pd
from collections import defaultdict
import statistics
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

brainFile = 'mini_project_2_data/brain_data_channel_one.csv'
eventsFile = 'mini_project_2_data/events_file_ordered.csv'

brainData = pd.read_csv(brainFile, header=None, names=['signal'])
eventsData = pd.read_csv(eventsFile, header=None, names=['startTime', 'peakTime', 'finger'], dtype=int)

'''
input:
    trial_points – a CSV file with three columns with the starting point of every movement, 
    the peak of every movement and the number finger (we worked on this together in the review session.) 
    Your function needs to make sure the data imported is of type “int” for ONLY this csv file.
    ------
    ecog_data – A CSV file of one column with the time series of the signal recorded using an ECOG electrode. 
    The indices in this file (number of rows) match up to the indices that appear in 
    the starting points and peak points in the trial_points data.
output:
    Your function will output a matrix with five rows and 1201 columns – 
    5x1201 into a variable named “fingers_erp_mean” in the order of fingers – 1, 2, 3, 4, 5.

'''
def calc_mean_erp(trial_points, ecog_data):
    # Dictionary to store signals corresponding to each finger
    # Using defaultdict to collect signal values at each sample point
    fingerDict = {
            1:defaultdict(list),
            2:defaultdict(list),
            3:defaultdict(list),
            4:defaultdict(list),
            5:defaultdict(list)
        }
    # Iterate through each row in the eventsData dataframe
    for index, row in eventsData.iterrows():
        finger = row['finger']  # Get the finger number for the event
        start = row['startTime'] - 200  # Define the start time window (200ms before event)
        end = row['startTime'] + 1000  # Define the end time window (1000ms after event)

        # Extract signal data for the given time window
        eventSample = list(brainData.loc[start:end, 'signal'])

        # Store signals for each time sample in the finger dictionary
        for i, signal in enumerate(eventSample):
            fingerDict[finger][i].append(signal)

    # Dictionary to store the mean ERP per finger for each sample
    fingerDictAverage = {
            1:defaultdict(int),
            2:defaultdict(int),
            3:defaultdict(int),
            4:defaultdict(int),
            5:defaultdict(int)
        }
    
    # Compute the mean ERP for each sample point across trials
    for finger in fingerDict:
        for sample in fingerDict[finger]:
            fingerDictAverage[finger][sample] = statistics.mean(fingerDict[finger][sample])
    
    # Assuming 'result' is the DataFrame with 5 rows and 1201 columns (fingers_erp_mean)
    def plot_finger_movements(fingers_erp_mean):
        time_points = range(-200, 1001)  # Time points from -200 ms to +1000 ms
        plt.figure(figsize=(12, 6))
        
        # Plot each finger's ERP
        for finger in range(1, 6):
            plt.plot(time_points, fingers_erp_mean.loc[finger], label=f'Finger {finger}')
        
        # Add title, labels, and legend
        plt.title('ERP Movements for Each Finger')
        plt.xlabel('Time (ms)')
        plt.ylabel('Mean Signal Amplitude')
        plt.legend(title='Finger')
        plt.grid(True)
        
        # Display the plot
        plt.show()

    # Call the function with the ERP data
    plot_finger_movements(pd.DataFrame(fingerDictAverage).T)

    # Convert the results into a pandas DataFrame for easy analysis
    return pd.DataFrame(fingerDictAverage).T

result = calc_mean_erp(eventsData, brainData)