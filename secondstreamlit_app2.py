import streamlit as st
import pandas as pd
st.set_page_config(layout="wide")
st.sidebar.image("https://static.wixstatic.com/media/abeab8_c2c418f5d500490e946bcb02ef2aa277~mv2.png/v1/crop/x_0,y_181,w_8672,h_2926/fill/w_304,h_103,al_c,q_85,usm_0.66_1.00_0.01,enc_auto/Summit%20Nanotech%20logo%20Transp.png", use_column_width=True)



def intro():
    import streamlit as st
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import csv
    import datetime
    import pandas as pd

  
    st.write("# Weekly Pilot Data Review")
    st.sidebar.success("Select an analysis track")

    st.markdown(
        """
This information is strictly confidential to Summit Nanotech.

App built in `Python 3.11` + `Streamlit` by Sean G [Analytics Overlord] + [Sustainability Grand Emperor]

    """
    )

# Upload CSV data
with st.sidebar.header('1. Upload weekly CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv)
""")
    dataset = pd.read_csv(uploaded_file)
    dataset = pd.read_csv(r"C:\Users\seanh\Downloads\HistorianData_DenaLiC_Extraction (11).csv")
    dataset['t_stamp'] = pd.to_datetime(dataset['t_stamp'])
   # dataset.set_index('t_stamp')


def mapping_demo():
    import streamlit as st
    import pandas as pd
    import pydeck as pdk

    from urllib.error import URLError

    st.markdown(f"# {list(page_names_to_funcs.keys())[2]}")
    st.write(
        """
        This demo shows how to use
[`st.pydeck_chart`](https://docs.streamlit.io/library/api-reference/charts/st.pydeck_chart)
to display geospatial data.
"""
    )

    @st.cache
    def from_data_file(filename):
        url = (
            "http://raw.githubusercontent.com/streamlit/"
            "example-data/master/hello/v1/%s" % filename
        )
        return pd.read_json(url)

    try:
        ALL_LAYERS = {
            "Bike Rentals": pdk.Layer(
                "HexagonLayer",
                data=from_data_file("bike_rental_stats.json"),
                get_position=["lon", "lat"],
                radius=200,
                elevation_scale=4,
                elevation_range=[0, 1000],
                extruded=True,
            ),
            "Bart Stop Exits": pdk.Layer(
                "ScatterplotLayer",
                data=from_data_file("bart_stop_stats.json"),
                get_position=["lon", "lat"],
                get_color=[200, 30, 0, 160],
                get_radius="[exits]",
                radius_scale=0.05,
            ),
            "Bart Stop Names": pdk.Layer(
                "TextLayer",
                data=from_data_file("bart_stop_stats.json"),
                get_position=["lon", "lat"],
                get_text="name",
                get_color=[0, 0, 0, 200],
                get_size=15,
                get_alignment_baseline="'bottom'",
            ),
            "Outbound Flow": pdk.Layer(
                "ArcLayer",
                data=from_data_file("bart_path_stats.json"),
                get_source_position=["lon", "lat"],
                get_target_position=["lon2", "lat2"],
                get_source_color=[200, 30, 0, 160],
                get_target_color=[200, 30, 0, 160],
                auto_highlight=True,
                width_scale=0.0001,
                get_width="outbound",
                width_min_pixels=3,
                width_max_pixels=30,
            ),
        }
        st.sidebar.markdown("### Map Layers")
        selected_layers = [
            layer
            for layer_name, layer in ALL_LAYERS.items()
            if st.sidebar.checkbox(layer_name, True)
        ]
        if selected_layers:
            st.pydeck_chart(
                pdk.Deck(
                    map_style="mapbox://styles/mapbox/light-v9",
                    initial_view_state={
                        "latitude": 37.76,
                        "longitude": -122.4,
                        "zoom": 11,
                        "pitch": 50,
                    },
                    layers=selected_layers,
                )
            )
        else:
            st.error("Please choose at least one layer above.")
    except URLError as e:
        st.error(
            """
            **This demo requires internet access.**

            Connection error: %s
        """
            % e.reason
        )





"""-------------------------------------------------------------------------------------------------------------------------------------------------------------"""






def sampling_demo():
    import streamlit as st

    
    import time
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    st.markdown(f'# {list(page_names_to_funcs.keys())[1]}')
    st.write(
        """## Pressure Drops"""
    )

    dataset['PT-265-DELTA'] = dataset['PT-265A']-dataset['PT-265B']
    st.line_chart(dataset, x = 't_stamp', y = ('PT-265A', 'PT-265B'))
    st.line_chart(dataset, x = 't_stamp', y = 'PT-265-DELTA')
    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
    last_rows = np.random.randn(1, 1)
    print("This is last rows?")
    print("This is last rows?")

    print("This is last rows?")
    print("This is last rows?")
    print(last_rows)
    
###### Plotly Implementation

    import plotly.express as px
    import streamlit as st
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Define a custom colorscale")
        df = px.data.iris()
        df = dataset
        fig = px.line(
            df,
            x="t_stamp",
            y="PT-265A",
        #   color="red"
            #color_continuous_scale="reds",
        )
        fig.add_scatter(x = df['t_stamp'], y = df['PT-265B'])

        tab1, tab2 = st.tabs(["Streamlit theme (default)", "Plotly native theme"])
        with tab1:
            st.plotly_chart(fig, theme="streamlit", use_container_width=True)
        with tab2:
            st.plotly_chart(fig, theme=None, use_container_width=True)
    
    
    ### Streamlit Native Plot - Example using random data generation, Animated
    with col2:
        chart = st.line_chart(last_rows)
        for i in range(1, 101):
            new_rows = last_rows[-1, :] + np.random.randn(5, 1).cumsum(axis=0)
            status_text.text("%i%% Complete" % i)

            chart.add_rows(new_rows)
            progress_bar.progress(i)
            last_rows = new_rows
            time.sleep(0.05)
        
        k = 0
        first_row = []
        print(first_row)





   
   
    append_set_time = []
    append_set_fit125 = []
    firstdata_dict = dataset.iloc[[1]]

    print("This is head")
    print("This is head")
    print("This is head")

    print(firstdata_dict.head(1))



    import itertools

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import streamlit as st
    import streamlit.components.v1 as components

    drawing1= st.empty()

    def data_gen():
        for cnt in itertools.count():
            t = cnt / 10
            yield t, np.sin(2*np.pi*t) * np.exp(-t/10.)
            #the_plot.pyplot(plt)


    def init():
        ax.set_ylim(-1.1, 1.1)
        ax.set_xlim(0, 10)
        del xdata[:]
        del ydata[:]
        line.set_data(xdata, ydata)
        return line

    fig, ax = plt.subplots()
    #fig.patch.set_facecolor('black')
    plt.style.use('dark_background')
    line, = ax.plot([], [], lw=2)
    ax.grid()

    xdata, ydata = [], []


    #the_plot = st.pyplot(plt)


    def run(data):
        # update the data
        t, y = data
        xdata.append(t)
        ydata.append(y)
        xmin, xmax = ax.get_xlim()

        if t >= xmax:
            ax.set_xlim(xmin, 2*xmax)
            ax.figure.canvas.draw()
        line.set_data(xdata, ydata)
        #the_plot.pyplot(plt)
        #the_plot.write(ax)
        #the_plot.write(fig)
        #the_plot.write(plt)
        
        drawing1.pyplot(plt)

        return line,

    ani = animation.FuncAnimation(fig, run, data_gen, interval=10, init_func=init)
    #the_plot.pyplot(plt)  
    plt.show()
    #the_plot.pyplot(plt)
    components.html(ani.to_jshtml(), height=1000)



    # for i in dataset2['FIT-125']:
    #     if k == 0:
    #         k = k + 1
    #         time.sleep(1)
    #         pass
    #     elif k == 1:
    #         k = k + 1
    #         time.sleep(1)
    #         pass
    #     else:
    #         new_array = [[]]
    #         a = dataset.iloc[[k]]
    #         a.set_index('t_stamp')['FIT-125']

    #         new_array[0].append(a)
    #         print("New Array!!")
    #         print(new_array)
    #         chart2.add_rows(new_array)
    #         time.sleep(1)
    #         k = k + 1

    # st.button("Re-run")


def data_frame_demo():
    import streamlit as st
    import pandas as pd
    import altair as alt

    from urllib.error import URLError

    st.markdown(f"# {list(page_names_to_funcs.keys())[3]}")
    st.write(
        """
        This demo shows how to use `st.write` to visualize Pandas DataFrames.

(Data courtesy of the [UN Data Explorer](http://data.un.org/Explorer.aspx).)
"""
    )

    @st.cache
    def get_UN_data():
        AWS_BUCKET_URL = "http://streamlit-demo-data.s3-us-west-2.amazonaws.com"
        df = pd.read_csv(AWS_BUCKET_URL + "/agri.csv.gz")
        return df.set_index("Region")

    try:
        df = get_UN_data()
        countries = st.multiselect(
            "Choose countries", list(df.index), ["China", "United States of America"]
        )
        if not countries:
            st.error("Please select at least one country.")
        else:
            data = df.loc[countries]
            data /= 1000000.0
            st.write("### Gross Agricultural Production ($B)", data.sort_index())

            data = data.T.reset_index()
            data = pd.melt(data, id_vars=["index"]).rename(
                columns={"index": "year", "value": "Gross Agricultural Product ($B)"}
            )
            chart = (
                alt.Chart(data)
                .mark_area(opacity=0.3)
                .encode(
                    x="year:T",
                    y=alt.Y("Gross Agricultural Product ($B):Q", stack=None),
                    color="Region:N",
                )
            )
            st.altair_chart(chart, use_container_width=True)
    except URLError as e:
        st.error(
            """
            **This demo requires internet access.**

            Connection error: %s
        """
            % e.reason
        )

page_names_to_funcs = {
    "Home": intro,
    "Weekly Analytics Run": sampling_demo,
    "Mapping Demo": mapping_demo,
    "DataFrame Demo": data_frame_demo
}

demo_name = st.sidebar.selectbox("Select a Track", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()