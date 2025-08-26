# AsteRisk: Intelligent Near-Earth Object (NEO) Risk Analyzer

AsteRisk is an interactive data-driven application designed to analyze and visualize Near-Earth Objects (NEOs) using NASA’s NeoWS API. With the growing importance of planetary defense, AsteRisk combines machine learning and visualization to make asteroid risk analysis accessible and actionable.

## Features
- **Automated Data Fetching**: Retrieve NEO data for a user-specified date range via NASA’s NeoWS API.
- **Preprocessing**: Extract key features like velocity, estimated diameter, and closest approach distance.
- **Clustering**: Apply KMeans clustering to categorize NEOs into similar groups.
- **Anomaly Detection**: Use Isolation Forest to highlight unusual or potentially risky NEOs.
- **3D Visualizations**: Interactive Plotly-based visuals to explore clusters and anomalies.
- **Risk Score**: Composite scoring system to rank NEOs by potential risk.
- **Top 5 Risky NEOs Dashboard**: Highlights the most concerning objects.
- **Interactive Filtering**: Streamlit sliders for refining by velocity, distance, and diameter.

## Tech Stack
- **Python**
- **Pandas, NumPy** for data processing
- **Scikit-learn** for ML models (KMeans, Isolation Forest)
- **Plotly** for visualization
- **Streamlit** for the interactive dashboard
- **NASA NeoWS API** for real-time data

## Installation
```bash
git clone https://github.com/yourusername/asterisk.git
cd asterisk
pip install -r requirements.txt
```

## Usage
```bash
streamlit run app.py
```

## Example
1. Choose a date range for fetching NASA NEO data.
2. Explore clustered NEOs and anomaly highlights.
3. Filter results interactively and review the Top 5 risky objects.

## Future Enhancements
- Support for multiple anomaly detection models.
- Integration with live orbital simulations.
- Automated daily data fetching.




