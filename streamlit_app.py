import numpy as np
import pandas as pd
import streamlit as st
#from pandas_profiling import ProfileReport
#from pandas.core.groupby.groupby import DataError
#from streamlit_pandas_profiling import st_profile_report
# Import libraries for analysis
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import csv
import datetime
#pd.set_option('precision', 5)
from datetime import datetime, timedelta
pcaddyinterp = r"C:\Users\seanh\OneDrive - Summit Nanotech Corporation\HistorianData_DenaLiC_Interp.csv"
pcaddyraw = r"C:\Users\seanh\OneDrive - Summit Nanotech Corporation\HistorianData_DenaLiC_Raw.csv"

st.sidebar.image("https://static.wixstatic.com/media/abeab8_c2c418f5d500490e946bcb02ef2aa277~mv2.png/v1/crop/x_0,y_181,w_8672,h_2926/fill/w_304,h_103,al_c,q_85,usm_0.66_1.00_0.01,enc_auto/Summit%20Nanotech%20logo%20Transp.png", use_column_width=True)


# Web App Title
st.markdown('''
# **Weekly Pilot Data Review**

This information is strictly confidential to Summit Nanotech.

App built in `Python 3.11` + `Streamlit` by Sean G [Analytics Overlord] + [Sustainability Grand Emperor]

---
''')

# Upload CSV data
with st.sidebar.header('1. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv)
""")

# Pandas Profiling Report
if uploaded_file is not None:
    @st.cache
    def load_csv():
        csv = pd.read_csv(uploaded_file)
        return csv
    df = load_csv()
    #pr = ProfileReport(df, explorative=True)
    st.header('**Input DataFrame**')
    st.write(df.describe())
    st.write('---')
    st.header('**Pandas Profiling Report**')
    #st_profile_report(pr)
else:
    st.info('Awaiting for CSV file to be uploaded.')
    if st.button('Press to use Example Dataset'):
        # Example data
        @st.cache
        def load_data():
            a = pd.DataFrame(
                np.random.rand(100, 5),
                columns=['a', 'b', 'c', 'd', 'e']
            )
            return a
        df = load_data()
        #pr = ProfileReport(df, explorative=True)
        st.header('**Input DataFrame**')
        st.write(df)
        st.write('---')
        st.header('**Pandas Profiling Report**')
      #  st_profile_report(pr)


### Grab the dataset that has been downloaded off the server
dataset = pd.read_csv(pcaddyinterp)
dataset['t_stamp'] = pd.to_datetime(dataset['t_stamp'])

### High level description of the server
dataset.describe()
st.write(dataset)
#dataset.dtypes
### Grab the dataset that has been downloaded off the server again to account for differences in pandas dataframe
dataset = pd.read_csv(pcaddyinterp)
datasetraw = pd.read_csv(pcaddyraw)
dataset['t_stamp'] = pd.to_datetime(dataset['t_stamp'])
datasetraw['t_stamp'] = pd.to_datetime(datasetraw['t_stamp'])
### Check interpolation method to ensure there isn't a massive difference between raw and interp

fig = plt.figure(figsize=(15,5))
plt.title("Interpolated LAB-265-LI vs Time")
plt.xlabel("Time", size=14)
plt.ylabel("Conc (PPM)", size=14)
plt.scatter(dataset['t_stamp'], dataset['LAB-265-LI'])
st.pyplot(fig)


fig = plt.figure(figsize=(15,15))
plt.title("NON-Interpolated LAB-265-LI vs Time")
plt.xlabel("Time", size=14)
plt.ylabel("Conc (PPM)", size=14)
plt.scatter(datasetraw['t_stamp'], datasetraw['LAB-265-LI'])
st.pyplot(fig)


fig = plt.figure(figsize=(15,15))
plt.title("Interpolated Na-Li Ratio")
plt.xlabel("Time", size = 14)
plt.ylabel("Ratio", size = 14)
plt.plot(dataset['t_stamp'], (dataset['LAB-265-NA']/dataset['LAB-265-LI']))
st.pyplot(fig)


import mysql.connector
 
mydb = mysql.connector.connect(
    host = "20.63.105.141",
    user = "tableau",
    password = "$umm1tN0w!",
    database = "denali", 
    auth_plugin='mysql_native_password'
)
 
cursor = mydb.cursor()
 
# Show existing tables
cursor.execute("SHOW TABLES")
 
for x in cursor:
  st.write(x)
  print(x)