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
import pickle

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



selected_features = [
    'premature_mortality', 'median_household_income', 'pct_65_and_older', 
    'driving_alone_to_work', 'injury_deaths', 'adult_obesity', 'frequent_mental_distress', 
    'single_parent_households', 'air_pollution_particulate_matter', 'median_age', 
    'adult_smoking', 'mammography_screening', 'housing_cost_challenges', 
    'social_associations', 'excessive_drinking', 'insufficient_sleep', 'pct_under_18', 
    'high_school_graduation', 'ratio_of_pop_to_dentists', 'uninsured_adults', 
    'preventable_hospital_stays', 'homeownership', 'poverty', 
    'children_in_poverty', 'unemployment_rate'
]



# Custom CSS to enhance the UI
custom_css = """
<style>
    .stApp {
        background-color: #000000;
    }
    .main > div {
        padding-top: 2rem;
    }
    .stSlider > div > div > div {
        background-color: #4e79a700;
    }
    .stSlider > div > div > div > div {
        color: white;
    }
    .stSlider > div > div > div > div > div {
        background-color: transparent !important;
    }
    .stPlotlyChart {
        background-color: white;
        border-radius: 5px;
        box-shadow: 0 2px 5px 0 rgba(0,0,0,0.16);
    }
    .css-1d391kg {
        background-color: #ffffff;
        border-radius: 5px;
        padding: 1rem;
        box-shadow: 0 2px 5px 0 rgba(0,0,0,0.16);
    }
</style>
"""


@st.cache_resource
def load_model_and_scaler():
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(BASE_DIR, "models", "xgboost_best_selected_model.pkl")
        scaler_path = os.path.join(BASE_DIR, "models", "scaler.pkl")
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            st.warning("Model or scaler file not found.")
            return None, None
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model or scaler: {str(e)}")
        return None, None

# Load data and model at app startup
df = load_data()
model, scaler = load_model_and_scaler()


def predict_life_expectancy():
    global df, model, scaler  # Use global variables

    st.title("🧬 Life Expectancy Predictor")
    st.write("Explore how community health indicators affect life expectancy. Adjust the sliders to see real-time changes in the prediction.")
    
    # Get feature ranges from the dataset
    feature_ranges = get_feature_ranges()
    input_dict = {}
    
    st.sidebar.header("Community Health Indicators")
    for feature, (min_val, max_val) in feature_ranges.items():
        # Calculate a sensible default value (e.g., median)
        default_val = df[feature].median()
        input_dict[feature] = st.sidebar.slider(
            feature.replace('_', ' ').title(),
            min_value=float(min_val),
            max_value=float(max_val),
            value=float(default_val),
            key=feature
        )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Community Health Indicators Visualization")
        polar_area_chart = get_polar_area_chart(input_dict)
        st.plotly_chart(polar_area_chart, use_container_width=True)
    
    with col2:
        st.subheader("Life Expectancy Prediction")
        if st.button("Predict Life Expectancy", key="predict_button", help="Click to calculate the predicted life expectancy based on current inputs"):
            input_array = np.array([input_dict[feature] for feature in selected_features]).reshape(1, -1)
            scaled_input = scaler.transform(input_array)
            prediction = model.predict(scaled_input)[0]
            st.markdown(f"<h1 style='text-align: center; color: #4e79a7;'>{prediction:.2f} years</h1>", unsafe_allow_html=True)
            st.info("This prediction is based on community health indicators and should not be used as a substitute for professional medical advice.")
        
        st.markdown("### How it works")
        st.write("1. Adjust the sliders in the sidebar to input community health indicators.")
        st.write("2. The polar area chart updates in real-time to visualize your inputs.")
        st.write("3. Click 'Predict Life Expectancy' to see the estimated life expectancy based on your inputs.")
        st.write("4. Experiment with different combinations to see how various factors affect life expectancy.")

def get_polar_area_chart(input_data):
    feature_ranges = get_feature_ranges()
    normalized_data = {}
    for feature, value in input_data.items():
        min_val, max_val = feature_ranges[feature]
        normalized_data[feature] = (value - min_val) / (max_val - min_val)

    categories = list(normalized_data.keys())
    values = list(normalized_data.values())
    
    fig = go.Figure(go.Barpolar(
        r=values,
        theta=categories,
        opacity=0.8,
        marker=dict(
            color=values,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='Normalized Value')
        )
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1]),
            angularaxis=dict(direction="clockwise")
        ),
        showlegend=False,
        height=600,
        margin=dict(l=80, r=80, t=20, b=20),
        title="Community Health Indicators"
    )
    return fig


def get_feature_ranges():
    global df  # Use global df
    feature_ranges = {}
    for feature in selected_features:
        min_val = df[feature].min()
        max_val = df[feature].max()
        feature_ranges[feature] = (min_val, max_val)
    return feature_ranges


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
    page = st.sidebar.radio("Select a page", ["Overview", "Life Expectancy Insights", "Disparities and Impact", "Life Expectancy Predictor"])
    
    if page == "Life Expectancy Predictor":
        pass
    elif page == "Overview":
        # Your existing Overview code
        pass
    elif page == "Life Expectancy Insights":
        # Your existing Insights code
        pass
    elif page == "Disparities and Impact":
        # Your existing Disparities code
        pass
    
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
            # Ensure we have valid data for all points
            mean_lons = []
            mean_lats = []
            mean_texts = []
            mean_colors = []
            
            median_lons = []
            median_lats = []
            median_texts = []
            median_colors = []

            for _, row in stats_df.iterrows():
                # Mean data
                mean_county_data = filtered_df[filtered_df['geo_name'] == row['mean_county']]
                if not mean_county_data.empty:
                    mean_lons.append(mean_county_data['longitude'].values[0])
                    mean_lats.append(mean_county_data['latitude'].values[0])
                    mean_texts.append(f"State: {row['state']}<br>Mean: {row['mean']:.2f}<br>County: {row['mean_county']}")
                    mean_colors.append(row['mean'])

                # Median data
                median_county_data = filtered_df[filtered_df['geo_name'] == row['median_county']]
                if not median_county_data.empty:
                    median_lons.append(median_county_data['longitude'].values[0])
                    median_lats.append(median_county_data['latitude'].values[0])
                    median_texts.append(f"State: {row['state']}<br>Median: {row['median']:.2f}<br>County: {row['median_county']}")
                    median_colors.append(row['median'])

            # Create traces only if we have data
            data = []
            
            if mean_lons:
                mean_data = go.Scattergeo(
                    lon=mean_lons,
                    lat=mean_lats,
                    text=mean_texts,
                    mode='markers',
                    name='Mean Life Expectancy',
                    marker=dict(
                        size=10,
                        color=mean_colors,
                        colorscale='RdYlGn',
                        colorbar=dict(
                            title='Life Expectancy (years)',
                            x=1.1  # Adjust position of colorbar
                        ),
                        cmin=min(stats_df['mean'].min(), stats_df['median'].min()),
                        cmax=max(stats_df['mean'].max(), stats_df['median'].max()),
                        symbol='circle'
                    ),
                    hoverinfo='text'
                )
                data.append(mean_data)

            if median_lons:
                median_data = go.Scattergeo(
                    lon=median_lons,
                    lat=median_lats,
                    text=median_texts,
                    mode='markers',
                    name='Median Life Expectancy',
                    marker=dict(
                        size=10,
                        color=median_colors,
                        colorscale='RdYlGn',
                        cmin=min(stats_df['mean'].min(), stats_df['median'].min()),
                        cmax=max(stats_df['mean'].max(), stats_df['median'].max()),
                        symbol='diamond'
                    ),
                    hoverinfo='text'
                )
                data.append(median_data)

            # Create the layout
            layout = go.Layout(
                geo=dict(
                    scope='usa',
                    projection_type='albers usa',
                    showland=True,
                    landcolor='rgb(20, 20, 20)',
                    countrycolor='rgb(40, 40, 40)',
                    showlakes=True,
                    lakecolor='rgb(20, 20, 20)',
                    subunitcolor='rgb(40, 40, 40)'
                ),
                paper_bgcolor='rgb(10, 10, 10)',
                plot_bgcolor='rgb(10, 10, 10)',
                margin=dict(l=0, r=0, t=0, b=0),
                showlegend=True,
                legend=dict(
                    x=0,
                    y=1,
                    bgcolor='rgba(0,0,0,0.5)',
                    font=dict(color='white')
                )
            )

            # Create the figure
            fig = go.Figure(data=data, layout=layout)
            
            # Update layout for dark theme compatibility
            fig.update_layout(
                height=600,
                font=dict(color='white'),
            )

            # Display the map
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Legend")
            
            # Explanatory text
            st.markdown("""
            The map shows two key statistics for each state:
            - **Circles**: Mean Life Expectancy
            - **Diamonds**: Median Life Expectancy
            
            Colors indicate life expectancy values:
            - Red: Lower life expectancy
            - Yellow: Medium life expectancy
            - Green: Higher life expectancy
            
            Hover over points to see detailed information.
            """)
            
            # Display summary statistics
            st.write("### Summary Statistics")
            st.write(f"Lowest Life Expectancy: {min(stats_df['mean'].min(), stats_df['median'].min()):.1f} years")
            st.write(f"Highest Life Expectancy: {max(stats_df['mean'].max(), stats_df['median'].max()):.1f} years")
            st.write(f"Average Life Expectancy: {((stats_df['mean'].mean() + stats_df['median'].mean()) / 2):.1f} years")
            
            # Add filter information
            if selected_state != "All":
                st.write(f"### Current Filter")
                st.write(f"State: {selected_state}")
            st.write(f"Year: {selected_year}")

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

    elif page == "Life Expectancy Predictor": 
        predict_life_expectancy()
        


if __name__ == "__main__":
    main()
        