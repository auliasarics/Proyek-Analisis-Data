import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.title("Analisis Kualitas Udara di 12 Wilayah")

combined_df = pd.read_csv('data_air_quality.csv.gz', compression='gzip')

st.subheader("Data Kualitas Udara yang Telah Diolah")
st.write(combined_df.head())

st.subheader("1. Rata-Rata PM2.5 per Wilayah")
pm25_avg = combined_df.groupby('station')['PM2.5'].mean().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x=pm25_avg.index, y=pm25_avg.values, hue=pm25_avg.index, palette="viridis", legend=False, ax=ax)
ax.set_title('Rata-Rata PM2.5 per Wilayah')
ax.set_xlabel('Wilayah')
ax.set_ylabel('Rata-Rata PM2.5')
plt.xticks(rotation=45)
st.pyplot(fig)

st.subheader("2. Korelasi antara PM2.5 dan Faktor Lainnya")
correlation_matrix = combined_df[['PM2.5', 'PM10', 'CO', 'SO2', 'NO2', 'O3', 'TEMP']].corr()

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
ax.set_title('Korelasi antara PM2.5 dan Faktor Lainnya')
st.pyplot(fig)

st.subheader("3. Perubahan PM2.5 Sepanjang Tahun")
combined_df['datetime'] = pd.to_datetime(combined_df[['year', 'month', 'day', 'hour']])
combined_df.set_index('datetime', inplace=True)

# Resampling data bulanan untuk PM2.5
monthly_pm25 = combined_df.groupby('station')['PM2.5'].resample('ME').mean().unstack(level=0)

# Visualisasi perubahan PM2.5 sepanjang tahun
fig, ax = plt.subplots(figsize=(15, 10))
for station in monthly_pm25.columns:
    ax.plot(monthly_pm25.index, monthly_pm25[station], label=station)
ax.set_title('Monthly Average PM2.5 Concentration by Station (2013-2017)')
ax.set_xlabel('Date')
ax.set_ylabel('PM2.5 Concentration')
ax.legend(title='Station', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(True)
plt.tight_layout()
st.pyplot(fig)

st.subheader("4. Clustering Stasiun Berdasarkan Kualitas Udara")
cluster_data = combined_df.groupby('station').agg({
    'PM2.5': 'mean',
    'PM10': 'mean',
    'SO2': 'mean',
    'NO2': 'mean',
    'CO': 'mean',
    'O3': 'mean'
}).reset_index()

scaler = StandardScaler()
cluster_data_scaled = scaler.fit_transform(cluster_data.drop(columns=['station']))

kmeans = KMeans(n_clusters=3, random_state=42)
cluster_data['cluster'] = kmeans.fit_predict(cluster_data_scaled)

fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x='PM2.5', y='PM10', hue='cluster', data=cluster_data, palette='viridis', s=100, ax=ax)
ax.set_title('Clustering Stasiun Berdasarkan Kualitas Udara')
ax.set_xlabel('Rata-Rata PM2.5')
ax.set_ylabel('Rata-Rata PM10')
ax.legend(title='Cluster')
st.pyplot(fig)