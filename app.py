import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy import stats
import base64
import folium
from streamlit_folium import folium_static
from branca.colormap import LinearColormap
import os


# Set page config
st.set_page_config(page_title="U.S. Life Expectancy Explorer", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "data", "chr_census_featured_engineered.csv")
video_path = os.path.join(BASE_DIR, "assets", "bluezonevideo.mp4")


# Load the data
@st.cache_data
def load_data():
    df = pd.read_csv(data_path)
    return df

df = load_data()


# Function to encode video file to base64
def get_base64_video(video_file):
    with open(video_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Function to set video as background
def set_video_background(video_file):
    video_base64 = get_base64_video(video_file)
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url(data:video/mp4;base64,{video_base64});
            background-size: cover;
        }}
        #myVideo {{
            position: fixed;
            right: 0;
            bottom: 0;
            min-width: 100%;
            min-height: 100%;
            width: auto;
            height: auto;
            z-index: -1;
            object-fit: cover;
        }}
        </style>
        <video autoplay muted loop id="myVideo">
            <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
        </video>
        """,
        unsafe_allow_html=True
    )


# Main function to run the app
def main():
    st.sidebar.title("Navigation")
    # Sidebar for page selection
    page = st.sidebar.radio("Select a page", ["Overview", "Life Expectancy Insights", "Disparities and Impact"])
    st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background-color: rgba(255, 255, 255, 0); /* Set transparency */
    }
    </style>
    """,
    unsafe_allow_html=True
    )
    if page == "Overview":
        
        set_video_background(video_path)

        # Title and introduction
        st.title("Exploring Life Expectancy in the U.S.")
        st.write("""
        This app explores factors influencing life expectancy across the United States, 
        with a focus on disparities between rural and urban areas. Key factors include 
        healthcare access, environmental conditions, lifestyle choices, and socioeconomic status.
        """)

        # Sidebar filters
        st.sidebar.header("Filters")
        selected_state = st.sidebar.selectbox("Select a State", ["All"] + sorted(df['state'].unique()))
        selected_year = st.sidebar.slider("Select a Year", int(df['year'].min()), int(df['year'].max()), int(df['year'].max()))

        # Filter data based on selections
        if selected_state != "All":
            filtered_df = df[(df['state'] == selected_state) & (df['year'] == selected_year)]
        else:
            filtered_df = df[df['year'] == selected_year]

        # Calculate national average life expectancy
        national_avg = filtered_df['life_expectancy'].mean()

        st.subheader("Life Expectancy Across U.S. Counties")
                
        # Display key metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("National Average Life Expectancy", f"{national_avg:.1f} years")
        col2.metric("Highest Life Expectancy", f"{filtered_df['life_expectancy'].max():.1f} years")
        col3.metric("Lowest Life Expectancy", f"{filtered_df['life_expectancy'].min():.1f} years")

        # Create map
        fig = px.scatter_mapbox(filtered_df, 
                                lat="latitude", 
                                lon="longitude", 
                                color="life_expectancy",
                                hover_name="geo_name",
                                zoom=3, 
                                mapbox_style="carto-positron",
                                color_continuous_scale="Viridis",
                                size_max=18,
                                width=1200,
                                height=800)
        # Set the background to be transparent
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',  
            plot_bgcolor='rgba(0,0,0,0)'    
        )

        st.plotly_chart(fig)

    elif page == "Life Expectancy Insights":
        st.title("Life Expectancy Insights")

        # Sidebar filters
        st.sidebar.header("Filters")
        selected_state = st.sidebar.selectbox("Select a State", ["All"] + sorted(df['state'].unique()))
        selected_year = st.sidebar.slider("Select a Year", int(df['year'].min()), int(df['year'].max()), int(df['year'].max()))

        # Filter data based on selections
        if selected_state != "All":
            filtered_df = df[(df['state'] == selected_state) & (df['year'] == selected_year)]
        else:
            filtered_df = df[df['year'] == selected_year]

        # Statistical Summaries
        st.header("Statistical Summaries")
        
        # Calculate statistics
        stats_df = filtered_df.groupby('state')['life_expectancy'].agg(['mean', 'median']).reset_index()
        
        # Find counties closest to each statistic
        for stat in ['mean', 'median']:
            stats_df[f'{stat}_county'] = stats_df.apply(lambda row: 
                filtered_df[filtered_df['state'] == row['state']].iloc[
                    (filtered_df[filtered_df['state'] == row['state']]['life_expectancy'] - row[stat]).abs().argsort()[:1]
                ]['geo_name'].values[0], axis=1)

        # Display statistics
        st.write(stats_df)

        # Map visualization
        st.subheader("Statistical Summaries Map")

        # Create two columns: one for the map and one for the legend
        col1, col2 = st.columns([3, 1])

        with col1:
            # Create a base map
            m = folium.Map(location=[37.0902, -95.7129], zoom_start=4, tiles="cartodbdark_matter")
            colormap = LinearColormap(colors=['red', 'yellow', 'green'], vmin=stats_df['mean'].min(), vmax=stats_df['mean'].max())

            # Add markers for mean and median
            for _, row in stats_df.iterrows():
                # Mean marker (circle)
                folium.CircleMarker(
                    location=[filtered_df[filtered_df['geo_name'] == row['mean_county']]['latitude'].values[0],
                            filtered_df[filtered_df['geo_name'] == row['mean_county']]['longitude'].values[0]],
                    radius=6,
                    popup=f"State: {row['state']}<br>Mean: {row['mean']:.2f}<br>County: {row['mean_county']}",
                    color='blue',
                    fill=True,
                    fillColor='blue',
                    fillOpacity=0.7
                ).add_to(m)

                # Median marker (diamond)
                folium.RegularPolygonMarker(
                    location=[filtered_df[filtered_df['geo_name'] == row['median_county']]['latitude'].values[0],
                            filtered_df[filtered_df['geo_name'] == row['median_county']]['longitude'].values[0]],
                    number_of_sides=4,
                    radius=5,
                    rotation=45,
                    popup=f"State: {row['state']}<br>Median: {row['median']:.2f}<br>County: {row['median_county']}",
                    color='red',
                    fill=True,
                    fillColor='red',
                    fillOpacity=0.7
                ).add_to(m)

            # Display the map
            folium_static(m)

        with col2:
            st.subheader("Legend")
            
            # Mean marker legend
            st.markdown(
                '<div style="display: flex; align-items: center; margin-bottom: 10px;">'
                '<div style="width: 20px; height: 20px; border-radius: 50%; background-color: blue; margin-right: 10px;"></div>'
                '<span>Mean Life Expectancy</span>'
                '</div>',
                unsafe_allow_html=True
            )
            
            # Median marker legend
            st.markdown(
                '<div style="display: flex; align-items: center; margin-bottom: 10px;">'
                '<div style="width: 20px; height: 20px; transform: rotate(45deg); background-color: red; margin-right: 10px;"></div>'
                '<span>Median Life Expectancy</span>'
                '</div>',
                unsafe_allow_html=True
            )
            
            # Colormap legend
            st.write("Life Expectancy (years)")
            colormap_html = f'''
            <div style="display: flex; flex-direction: column; align-items: stretch;">
                <div style="height: 20px; background: linear-gradient(to right, red, yellow, green);"></div>
                <div style="display: flex; justify-content: space-between;">
                    <span>{stats_df['mean'].min():.1f}</span>
                    <span>{stats_df['mean'].max():.1f}</span>
                </div>
            </div>
            '''
            st.markdown(colormap_html, unsafe_allow_html=True)

        # Correlation and Regression Analysis
        st.header("Correlation and Regression Analysis")
        
        factors = ['access_to_exercise_opportunities', 'adult_obesity', 'air_pollution_particulate_matter',
                'children_in_poverty', 'drinking_water_violations', 'excessive_drinking',
                'frequent_mental_distress', 'high_school_graduation', 'income_inequality',
                'median_household_income', 'poverty', 'ratio_of_pop_to_pcp', 'unemployment_rate', 'uninsured_adults']
        
        selected_factors = st.multiselect("Select factors for correlation analysis", factors, default=factors)
        
        corr_matrix = filtered_df[selected_factors + ['life_expectancy']].corr()
        fig_heatmap = px.imshow(corr_matrix, 
                                labels=dict(color="Correlation"),
                                x=selected_factors + ['life_expectancy'],
                                y=selected_factors + ['life_expectancy'],
                                color_continuous_scale="RdBu_r")
        fig_heatmap.update_layout(title="Correlation Heatmap", width=800, height=600)
        st.plotly_chart(fig_heatmap)

        # Regression analysis
        st.subheader("Regression Analysis")
        selected_factor_regression = st.selectbox("Select a factor for regression analysis", factors)
        
        x = filtered_df[selected_factor_regression]
        y = filtered_df['life_expectancy']
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        fig_regression = px.scatter(filtered_df, x=selected_factor_regression, y='life_expectancy', trendline="ols")
        fig_regression.update_layout(title=f"Regression: {selected_factor_regression} vs Life Expectancy", 
                                    width=1000, height=700)
        st.plotly_chart(fig_regression)
        
        st.write(f"R-squared: {r_value**2:.4f}")    
        st.write(f"p-value: {p_value:.4f}")

        # Storyline conclusions
        st.header("Key Insights")
        st.write("""
        - Statistical summaries show variations in life expectancy across states.
        - The map highlights states with the highest and lowest life expectancy statistics.
        - Factors such as income, education, and healthcare access tend to have strong correlations with life expectancy.
        - The impact of various factors on life expectancy can vary significantly between states and counties.
        """)
    
    elif page == "Disparities and Impact":
        st.title("Understanding Disparities")

        # Sidebar filters
        st.sidebar.header("Filters")
        selected_state = st.sidebar.selectbox("Select a State", ["All"] + sorted(df['state'].unique()))
        selected_year = st.sidebar.slider("Select a Year", int(df['year'].min()), int(df['year'].max()), int(df['year'].max()))
        population_threshold = st.sidebar.slider("Population threshold for Urban classification", 10000, 100000, 50000)

        # Filter data based on selections
        if selected_state != "All":
            filtered_df = df[(df['state'] == selected_state) & (df['year'] == selected_year)]
        else:
            filtered_df = df[df['year'] == selected_year]
        filtered_df['area_type'] = np.where(filtered_df['population'] > population_threshold, 'Urban', 'Rural')

        # List of factors to analyze
        factors = ['income_inequality', 'high_school_graduation', 'adult_obesity', 'access_to_exercise_opportunities', 
                'air_pollution_particulate_matter', 'children_in_poverty', 'drinking_water_violations', 
                'excessive_drinking', 'frequent_mental_distress', 'median_household_income', 'poverty', 
                'ratio_of_pop_to_pcp', 'unemployment_rate', 'uninsured_adults', 'life_expectancy']

        # Feature Map
        st.header("Feature Distribution Map")
        selected_feature = st.selectbox("Select a feature to visualize", factors)
        
        # Add data checks and logging
        st.write(f"Data range for {selected_feature}: {filtered_df[selected_feature].min()} to {filtered_df[selected_feature].max()}")
        st.write(f"Number of non-null values: {filtered_df[selected_feature].count()}")
        
        # Ensure the selected feature has numeric data
        if pd.api.types.is_numeric_dtype(filtered_df[selected_feature]):
            fig = px.scatter_mapbox(filtered_df, 
                                    lat="latitude", 
                                    lon="longitude", 
                                    color=selected_feature,
                                    size=selected_feature,
                                    size_max=10,
                                    zoom=3, 
                                    center={"lat": 37.0902, "lon": -95.7129},
                                    mapbox_style="carto-positron",
                                    hover_name="geo_name",
                                    hover_data=[selected_feature, 'area_type'],
                                    color_continuous_scale="Viridis",
                                    title=f"Distribution of {selected_feature} Across Counties")
            
            fig.update_layout(height=600, margin={"r":0,"t":30,"l":0,"b":0})
            st.plotly_chart(fig)
        else:
            st.error(f"The selected feature '{selected_feature}' is not numeric. Please choose a numeric feature for visualization.")

        # Rural vs. Urban Disparities
        st.header("Rural vs. Urban Disparities")
        
        fig = go.Figure()
        for area in ['Rural', 'Urban']:
            fig.add_trace(go.Box(y=filtered_df[filtered_df['area_type'] == area][selected_feature], name=area))
        fig.update_layout(title=f"{selected_feature} in Rural vs Urban Areas", width=800, height=500)
        st.plotly_chart(fig)

        # Case Study Analysis
        st.header("Case Study Analysis")
        
        # Find counties with extreme values for the selected feature
        low_county = filtered_df.loc[filtered_df[selected_feature].idxmin()]
        high_county = filtered_df.loc[filtered_df[selected_feature].idxmax()]

        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"Lowest {selected_feature}")
            st.write(f"County: {low_county['geo_name']}")
            st.write(f"{selected_feature}: {low_county[selected_feature]:.2f}")
            st.write("Other Key Factors:")
            for factor in [f for f in factors if f != selected_feature][:5]:  # Display top 5 other factors
                st.write(f"- {factor}: {low_county[factor]:.2f}")

        with col2:
            st.subheader(f"Highest {selected_feature}")
            st.write(f"County: {high_county['geo_name']}")
            st.write(f"{selected_feature}: {high_county[selected_feature]:.2f}")
            st.write("Other Key Factors:")
            for factor in [f for f in factors if f != selected_feature][:5]:  # Display top 5 other factors
                st.write(f"- {factor}: {high_county[factor]:.2f}")

        # Heatmap Visualization
        st.header("Feature Relationships Heatmap")
        
        correlation_data = filtered_df[factors].corr()[selected_feature].sort_values(ascending=False)
        
        fig = px.imshow(correlation_data.values.reshape(-1, 1),
                        y=correlation_data.index,
                        color_continuous_scale="RdBu_r",
                        labels=dict(color="Correlation"),
                        title=f"Correlation of {selected_feature} with Other Features")
        fig.update_layout(width=800, height=600)
        st.plotly_chart(fig)

        # Recommendations
        st.header("Recommendations")
        st.write("""
        Based on our analysis, here are some data-driven recommendations to improve life expectancy:
        1. Improve healthcare access in rural areas
        2. Invest in education and job opportunities to reduce poverty
        3. Implement environmental policies to reduce air pollution
        4. Promote healthy lifestyle choices and access to exercise opportunities
        5. Address mental health issues and provide better access to mental health services
        """)

        # Key Insights
        st.header("Key Insights")
        st.write(f"""
        - The map reveals significant geographic variations in {selected_feature} across counties.
        - Rural and urban areas show distinct patterns in {selected_feature} distribution.
        - Extreme cases highlight the range of disparities and potential areas for targeted interventions.
        - The heatmap reveals strong correlations between certain factors and {selected_feature}, which can guide policy decisions.
        - Addressing these disparities requires a multi-faceted approach, considering local contexts and needs.
        - Policy interventions should be tailored to the specific challenges faced by each community, as the impact of various factors can differ between regions.
        """)


if __name__ == "__main__":
    main()
        