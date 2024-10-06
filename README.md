# U.S. Life Expectancy Explorer

This Streamlit web application explores life expectancy across U.S. counties, highlighting disparities between rural and urban areas. The app allows users to visualize factors influencing life expectancy, including healthcare access, environmental conditions, and socioeconomic factors, with interactive maps and statistical summaries.

![App Preview](preview_image.png)

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Technologies Used](#technologies-used)
6. [Data Source](#data-source)
7. [Contributing](#contributing)
8. [License](#license)

## Overview

The **U.S. Life Expectancy Explorer** allows users to analyze life expectancy trends across U.S. counties. It provides insights into disparities between urban and rural areas, examining factors such as healthcare, obesity rates, air pollution, and socioeconomic status. Users can explore these factors interactively through maps, statistical summaries, and regression analysis.

## Features

- **Life Expectancy Map**: Visualizes life expectancy across U.S. counties using a scatter map with interactive filtering by state and year.
- **Statistical Summaries**: Provides detailed statistics such as mean and median life expectancy, along with the counties closest to these values.
- **Correlation and Regression Analysis**: Users can analyze the relationship between various factors and life expectancy through correlation matrices and regression plots.
- **Video Background**: The app features an engaging video background for enhanced user experience.
  
## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/life-expectancy-explorer.git
    cd life-expectancy-explorer
    ```

2. **Install the dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Download the dataset**:
   Place the `chr_census_featured_engineered.csv` dataset inside the `data` folder. The dataset should contain life expectancy data across U.S. counties.

4. **Add assets**:
   Place the video file (`bluezonevideo.mp4`) in the `assets` folder to be used as the app's background.

## Usage

Run the following command to start the Streamlit app locally:

```bash
streamlit run app.py
```

Once the server starts, open your browser and navigate to `http://localhost:8501` to explore the app.

### User Interaction

- Navigate between pages using the sidebar.
- Use the state and year filters to customize the data displayed in the map and statistical sections.
- Analyze correlations between life expectancy and various socioeconomic or environmental factors using the correlation matrix and regression analysis tools.

## Technologies Used

- **Streamlit**: For building the interactive web app.
- **Pandas**: For data manipulation and analysis.
- **Plotly**: For creating interactive charts and maps.
- **Folium**: For geographic visualization.
- **Numpy** and **Scipy**: For statistical calculations.

## Data Source

The dataset used in this app is derived from the [County Health Rankings](https://www.countyhealthrankings.org/) and census data. It includes life expectancy and other health-related factors across U.S. counties.

## Contributing

Contributions are welcome! If you have suggestions or find bugs, feel free to create an issue or submit a pull request.

1. Fork the repository
2. Create a new branch: `git checkout -b my-feature-branch`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin my-feature-branch`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

