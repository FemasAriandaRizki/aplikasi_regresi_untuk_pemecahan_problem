# Nama   : Femas Arianda Rizki
# NIM    : 21120122130080
# Kelas  : Metode Numerik - B

# Kode Sumber
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Memuat data dari file CSV
data = pd.read_csv("Student_Performance.csv")

# Mengekstrak kolom "Hours Studied" dan "Performance Index"
TB = data['Hours Studied'].values
NT = data['Performance Index'].values

# Log-transformasi data
log_TB = np.log(TB)
log_NT = np.log(NT)

# Menyusun matriks untuk eliminasi Gauss
A = np.vstack([log_TB, np.ones(len(log_TB))]).T
b = log_NT

# Eliminasi Gauss untuk menyelesaikan Ax = b
coeffs = np.linalg.lstsq(A, b, rcond=None)[0]
b_est = coeffs[0]
log_C_est = coeffs[1]
C_est = np.exp(log_C_est)

# kode testing
print(f"Persamaan regresi pangkat sederhana: y = {C_est:.4f} * x^{b_est:.4f}")
x = 4.5
print(f'TB = {x}')
print(f'Untuk TB = {x}, akan menghasilkan NT kira-kira sebesar {(C_est * x**b_est):.4f}')

# Buat garis regresi dengan x yang lebih halus
x_fit = np.linspace(min(TB), max(TB), 100)
y_fit = C_est * x_fit**b_est

# Plot grafik data dan hasil regresi
plt.figure(figsize=(10, 6))
plt.plot(TB, NT, 'o', label='Original data', markersize=4)
plt.plot(x_fit, y_fit, color='red', label=f'Regresi: y = {C_est:.4f} * x^{b_est:.4f}')
plt.xlabel('Waktu belajar (TB)')
plt.ylabel('Nilai ujian siswa (NT)')
plt.title('Hubungan durasi waktu belajar (TB) terhadap nilai ujian siswa (NT)')
plt.legend()
plt.grid(True)
plt.show()

# Galat RMS
predicted_NT = C_est * TB**b_est
residuals = NT - predicted_NT
squared_residuals = residuals ** 2
mean_squared_error = np.mean(squared_residuals)
rms = np.sqrt(mean_squared_error)

print(f'Galat pada regresi metode pangkat sederhana ini yaitu: {rms:.6f}')