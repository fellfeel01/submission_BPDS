import streamlit as st
import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Fungsi untuk memuat dan memproses data dari database
def load_data():
    conn = sqlite3.connect('metabase.db.mv.db')  
    query = "SELECT * FROM Students"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

# Memuat data
df = load_data()

# Mengkodekan kolom target (Status) menjadi numerik
le = LabelEncoder()
df['Status'] = le.fit_transform(df['Status'])  # Dropout=0, Enrolled=1, Graduate=2

# Memilih fitur dan target
features = ['Age_at_enrollment', 'Admission_grade', 'Unemployment_rate', 'GDP', 'Application_mode']
X = df[features]
y = df['Status']

# Membagi data menjadi train dan test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Melatih model Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Judul aplikasi
st.title("Prediksi Status Mahasiswa")

# Input fitur dari pengguna
st.header("Masukkan Data Mahasiswa")
age = st.number_input("Usia saat Pendaftaran", min_value=0, max_value=100, value=20)
admission_grade = st.number_input("Nilai Admission", min_value=0.0, max_value=200.0, value=120.0)
unemployment_rate = st.number_input("Tingkat Pengangguran (%)", min_value=0.0, max_value=20.0, value=10.0)
gdp = st.number_input("GDP", min_value=-100.0, max_value=200.0, value=50.0)
application_mode = st.number_input("Mode Pendaftaran", min_value=0, max_value=100, value=53)

# Tombol untuk prediksi
if st.button("Prediksi Status"):
    # Membuat input array untuk prediksi
    input_data = [[age, admission_grade, unemployment_rate, gdp, application_mode]]
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    # Mengonversi prediksi kembali ke label asli
    status_mapping = {0: 'Dropout', 1: 'Enrolled', 2: 'Graduate'}
    predicted_status = status_mapping[prediction[0]]
    probabilities = {status_mapping[i]: prob * 100 for i, prob in enumerate(prediction_proba[0])}

    # Tampilkan hasil
    st.success(f"Status Prediksi: **{predicted_status}**")
    st.subheader("Probabilitas:")
    for status, prob in probabilities.items():
        st.write(f"{status}: {prob:.2f}%")

# Catatan untuk pengguna
st.info("Pastikan nilai yang dimasukkan sesuai dengan rentang data asli. Model ini menggunakan Random Forest dengan akurasi berdasarkan data pelatihan.")