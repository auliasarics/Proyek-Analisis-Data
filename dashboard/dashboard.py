import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap
import streamlit as st
from streamlit_folium import folium_static
import requests

# Judul Dashboard
st.title("Analisis Kualitas Udara di 12 Stasiun")

# Fungsi untuk mengunduh dan memuat data
@st.cache_data
def load_data():
    # URL file di GitHub
    url = "https://github.com/auliasarics/Proyek-Analisis-Data/raw/main/dashboard/data_air_quality.csv.gz"
    
    # Mengunduh file
    response = requests.get(url, stream=True)
    with open("data_air_quality.csv.gz", "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            f.write(chunk)
    
    # Membaca file yang sudah diunduh
    data = pd.read_csv("data_air_quality.csv.gz", compression="gzip")
    return data

# Memuat data
combined_df = load_data()

# Tampilkan data
st.header("Data Kualitas Udara")
st.write(combined_df.head())

# Visualisasi Rata-rata PM2.5 per Wilayah
st.header("Rata-Rata PM2.5 per Wilayah")
pm25_avg = combined_df.groupby('station')['PM2.5'].mean().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x=pm25_avg.index, y=pm25_avg.values, hue=pm25_avg.index, palette="viridis", legend=False, ax=ax)
ax.set_title('Rata-Rata PM2.5 per Wilayah')
ax.set_xlabel('Wilayah')
ax.set_ylabel('Rata-Rata PM2.5')
ax.tick_params(axis='x', rotation=45)
st.pyplot(fig)

# Visualisasi Korelasi antara PM2.5 dan Faktor Lainnya
st.header("Korelasi antara PM2.5 dan Faktor Lainnya")
correlation_matrix = combined_df[['PM2.5', 'PM10', 'CO', 'SO2', 'NO2', 'O3', 'TEMP']].corr()
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
ax.set_title('Korelasi antara PM2.5 dan Faktor Lainnya')
st.pyplot(fig)

# Visualisasi Pola PM2.5 Sepanjang Tahun
st.header("Pola PM2.5 Sepanjang Tahun")
combined_df['datetime'] = pd.to_datetime(combined_df[['year', 'month', 'day', 'hour']])
combined_df.set_index('datetime', inplace=True)
monthly_pm25 = combined_df.groupby('station')['PM2.5'].resample('ME').mean().unstack(level=0)
fig, ax = plt.subplots(figsize=(15, 10))
for station in monthly_pm25.columns:
    ax.plot(monthly_pm25.index, monthly_pm25[station], label=station)
ax.set_title('Monthly Average PM2.5 Concentration by Station (2013-2017)')
ax.set_xlabel('Date')
ax.set_ylabel('PM2.5 Concentration')
ax.legend(title='Station', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(True)
st.pyplot(fig)

# Peta 
st.header("Peta Interaktif Kualitas Udara")
station_coords = {
    "Aotizhongxin": [39.9042, 116.4074],
    "Changping": [40.2181, 116.2317],
    "Dingling": [40.2928, 116.2167],
    "Dongsi": [39.9289, 116.4167],
    "Guanyuan": [39.9333, 116.3667],
    "Gucheng": [39.9167, 116.1833],
    "Huairou": [40.3167, 116.6333],
    "Nongzhanguan": [39.9333, 116.4667],
    "Shunyi": [40.1333, 116.6667],
    "Tiantan": [39.8833, 116.4167],
    "Wanliu": [39.9833, 116.2833],
    "Wanshouxigong": [39.8833, 116.3667]
}

pm25_avg = combined_df.groupby('station')['PM2.5'].mean().reset_index()
pm25_avg['latitude'] = pm25_avg['station'].map({k: v[0] for k, v in station_coords.items()})
pm25_avg['longitude'] = pm25_avg['station'].map({k: v[1] for k, v in station_coords.items()})

m = folium.Map(location=[39.9042, 116.4074], zoom_start=10)
for idx, row in pm25_avg.iterrows():
    folium.Marker(
        location=[row['latitude'], row['longitude']],
        popup=f"{row['station']}: Avg PM2.5 = {row['PM2.5']:.2f}",
        tooltip=row['station']
    ).add_to(m)

heat_data = [[row['latitude'], row['longitude'], row['PM2.5']] for idx, row in pm25_avg.iterrows()]
HeatMap(heat_data).add_to(m)

folium_static(m)

