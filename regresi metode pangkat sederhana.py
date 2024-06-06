import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Membaca data dari file CSV
data = pd.read_csv("Student_Performance.csv")

# Ekstraksi kolom "Hours Studied" dan "Performance Index"
TB = data['Hours Studied'].values
NT = data['Performance Index'].values

# Transformasi logaritma pada data
ln_TB = np.log(TB)
ln_NT = np.log(NT)

# Jumlah data poin (n)
n = len(TB)

# Membangun matriks A dan vektor b untuk sistem persamaan linier Y = a + bX
A = np.vstack([ln_TB, np.ones(n)]).T
b = ln_NT

# Menyelesaikan persamaan Ax = b menggunakan metode kuadrat terkecil (least squares)
x = np.linalg.lstsq(A, b, rcond=None)[0]

# Memisahkan nilai a dan b dari solusi x
a, b = x

# Menghitung konstanta C sebagai e^a
C = np.exp(a)

# Mencetak persamaan regresi pangkat
print(f'Persamaan regresi pangkat sederhananya yaitu: y = {C:.4f} * x^{b:.4f}')
x = 4.5
y = C*(x**b)
print(f'TB = {x}')
print(f'Untuk TB = {x}, akan menghasilkan NT kira-kira sebesar {y:.4f}')

# Plot data asli dan hasil regresi
plt.figure(figsize=(10, 6))
plt.plot(TB, NT, 'o', label='Original data', markersize=4)
# plt.plot(TB, C * (TB**b), 'r', label='Fitted line')
plt.xlabel('Waktu belajar (TB)')
plt.ylabel('Nilai ujian siswa (NT)')
plt.title('Hubungan durasi waktu belajar (TB) terhadap nilai ujian siswa (NT)')
plt.legend()
plt.grid(True)
plt.show()

# Menghitung dan menampilkan galat RMS
# predicted_NT = C * TB**b
# residuals = NT - predicted_NT
# squared_residuals = residuals ** 2
# mean_squared_error = np.mean(squared_residuals)
# rms = np.sqrt(mean_squared_error)

# print(f'Galat pada regresi metode pangkat ini yaitu: {rms:.6f}')
