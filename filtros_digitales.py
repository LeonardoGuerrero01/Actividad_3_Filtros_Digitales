
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, firwin, filtfilt, freqz

# =====================================
# 1. CREAR SENAL ORIGINAL
# =====================================

fs = 1000
t = np.arange(0, 2, 1/fs)

senal_original = (
    np.sin(2 * np.pi * 5 * t)
    + 0.7 * np.sin(2 * np.pi * 50 * t)
    + 0.5 * np.sin(2 * np.pi * 150 * t)
)

# Agregar ruido blanco
np.random.seed(42)
ruido = 0.4 * np.random.randn(len(t))
senal_ruidosa = senal_original + ruido


# =====================================
# 2. DISENAR FILTROS IIR BUTTERWORTH
# =====================================

b_iir_bajos, a_iir_bajos = butter(
    4, 20, btype="low", fs=fs
)

b_iir_altos, a_iir_altos = butter(
    4, 80, btype="high", fs=fs
)

b_iir_banda, a_iir_banda = butter(
    4, [30, 70], btype="band", fs=fs
)


# =====================================
# 3. DISENAR FILTROS FIR
# =====================================

b_fir_bajos = firwin(
    101, 20, fs=fs, pass_zero="lowpass"
)

b_fir_altos = firwin(
    101, 80, fs=fs, pass_zero="highpass"
)

b_fir_banda = firwin(
    101, [30, 70], fs=fs, pass_zero="bandpass"
)


# =====================================
# 4. APLICAR LOS FILTROS
# =====================================

iir_bajos = filtfilt(
    b_iir_bajos, a_iir_bajos, senal_ruidosa
)

iir_altos = filtfilt(
    b_iir_altos, a_iir_altos, senal_ruidosa
)

iir_banda = filtfilt(
    b_iir_banda, a_iir_banda, senal_ruidosa
)

fir_bajos = filtfilt(
    b_fir_bajos, [1.0], senal_ruidosa
)

fir_altos = filtfilt(
    b_fir_altos, [1.0], senal_ruidosa
)

fir_banda = filtfilt(
    b_fir_banda, [1.0], senal_ruidosa
)


# =====================================
# 5. GRAFICAR RESULTADOS IIR
# =====================================

plt.figure(figsize=(12, 10))

plt.subplot(4, 1, 1)
plt.plot(t, senal_ruidosa)
plt.title("Senal original con ruido")
plt.ylabel("Amplitud")
plt.grid()

plt.subplot(4, 1, 2)
plt.plot(t, iir_bajos)
plt.title("Filtro IIR pasa bajos")
plt.ylabel("Amplitud")
plt.grid()

plt.subplot(4, 1, 3)
plt.plot(t, iir_altos)
plt.title("Filtro IIR pasa altos")
plt.ylabel("Amplitud")
plt.grid()

plt.subplot(4, 1, 4)
plt.plot(t, iir_banda)
plt.title("Filtro IIR pasa bandas")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.grid()

plt.tight_layout()
plt.savefig("resultados_IIR.png", dpi=300)
plt.close()


# =====================================
# 6. GRAFICAR RESULTADOS FIR
# =====================================

plt.figure(figsize=(12, 10))

plt.subplot(4, 1, 1)
plt.plot(t, senal_ruidosa)
plt.title("Senal original con ruido")
plt.ylabel("Amplitud")
plt.grid()

plt.subplot(4, 1, 2)
plt.plot(t, fir_bajos)
plt.title("Filtro FIR pasa bajos")
plt.ylabel("Amplitud")
plt.grid()

plt.subplot(4, 1, 3)
plt.plot(t, fir_altos)
plt.title("Filtro FIR pasa altos")
plt.ylabel("Amplitud")
plt.grid()

plt.subplot(4, 1, 4)
plt.plot(t, fir_banda)
plt.title("Filtro FIR pasa bandas")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.grid()

plt.tight_layout()
plt.savefig("resultados_FIR.png", dpi=300)
plt.close()


# =====================================
# 7. RESPUESTA EN FRECUENCIA
# =====================================

plt.figure(figsize=(12, 8))

filtros = [
    (b_iir_bajos, a_iir_bajos, "IIR pasa bajos"),
    (b_iir_altos, a_iir_altos, "IIR pasa altos"),
    (b_iir_banda, a_iir_banda, "IIR pasa bandas"),
    (b_fir_bajos, [1.0], "FIR pasa bajos"),
    (b_fir_altos, [1.0], "FIR pasa altos"),
    (b_fir_banda, [1.0], "FIR pasa bandas")
]

for b, a, nombre in filtros:

    frecuencias, respuesta = freqz(
        b, a, fs=fs
    )

    magnitud = 20 * np.log10(
        np.maximum(np.abs(respuesta), 1e-10)
    )

    plt.plot(
        frecuencias,
        magnitud,
        label=nombre
    )

plt.title("Respuesta en frecuencia de filtros FIR e IIR")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud (dB)")
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig("respuesta_frecuencia_completa.png", dpi=300)
plt.show()


# =====================================
# 8. FINALIZAR SIMULACION
# =====================================

print("Simulacion terminada correctamente.")
print("Se generaron las siguientes graficas:")
print("resultados_IIR.png")
print("resultados_FIR.png")
print("respuesta_frecuencia_completa.png")

