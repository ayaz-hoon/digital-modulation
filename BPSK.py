# AUTHOR: AIYAZ AHMED N.HANSBHANVI
#( 201104003, TE-ETC Engineering, SEM-5,2021-22,GEC )
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft ,fftshift 
font1 ={"color":"black", "size":18}
font2 ={"color":"black", "size":20}  # FONT SIZE
bit_seq = [1,1,0,1,0]     # BIT SEQUENCE
num_bits = len(bit_seq)   # NUMBER OF BITS
fs = 1000                  # SAMPLING FREQUENCY
fc = 10                   # CARRIER FREQUENCY
t = np.arange(0, 5,1/fs)  # TIME INDEX
L =  len(t)               # LENGTH OF TIME AXIS
f =np.arange(-L/2,L/2,1)*1/5  # FREQUENCY AXIS
Vc =2                     # PEAK AMPLITUDE OF  CARRIER SIGNAL
vc = Vc*np.sin(2*np.pi*fc*t)    # CARRIER SIGNAL
Tb = 5/num_bits       # BIT DURATION
# CREATING LIST OF -1 AND 1 OF BIT DURATION Tb
neg_ones = []
for i in range(1,L):
    if t[i] <= Tb:
        z = -1
        neg_ones.append(z)
        
ones = []
for j in range(1,L):
    if t[j]<=Tb:
        o = 1
        ones.append(o)
print("this is list of ones of bit duration tb: ",ones)
print("this lenght of ones in tb: ",len(ones))
# CREATING BIPOLAR SQUARE WAVE ACCORDING TO THE BIT SEQUENCE
bipolar_sqr = []
for j in range(len(bit_seq)):
    if bit_seq[j] == 0:
        bipolar_sqr = bipolar_sqr + neg_ones
        
    else:
        bipolar_sqr = bipolar_sqr + ones
# EXPRESION OF  PSK SIGNAL 
vpsk = vc*bipolar_sqr
v_spec = fftshift(fft(vpsk))  # TAKING FFT OF PSK SIGNAL

print("this is bipolar squarewave: ",bipolar_sqr)
print("this lenght of  bipolar squarewave: ",len( bipolar_sqr))
# # TIME DOMAIN PLOTS
# plt.figure(1)  
# plt.subplot(3,1,1)      
# plt.plot(t,bipolar_sqr)
# plt.grid()
# plt.xlabel("Time in seconds",fontdict=font1)
# plt.ylabel("Amplitude in volts",fontdict=font1)
# plt.title("Modulating Signal v/s Time",fontdict=font2)
# plt.legend(["Bits =(1,1,0,1,0)"],loc = "upper right")
# plt.tick_params(axis = "both",labelsize = 16)  # ADJUSTING TICKS FONTSIZE

# plt.subplot(3,1,3)
# plt.plot(t,vc)
# plt.grid()
# plt.xlabel("Time in seconds",fontdict=font1)
# plt.ylabel("Amplitude in volts",fontdict=font1)
# plt.title("Carrier Signal v/s Time",fontdict=font2)
# plt.tick_params(axis = "both",labelsize = 16)   # ADJUSTING TICKS FONTSIZE

# plt.figure(2)
# plt.subplot(3,1,1)
# plt.plot(t,vpsk)
# plt.grid()
# plt.xlabel("Time in seconds",fontdict=font1)
# plt.ylabel("Amplitude in volts",fontdict=font1)
# plt.title("PSK Signal v/s Time",fontdict=font2)
# plt.tick_params(axis = "both",labelsize = 16)   # ADJUSTING TICKS FONTSIZE
# # FREQUENCY DOMAIN PLOTS
# plt.subplot(3,1,3)
# plt.plot(f,abs(v_spec))
# plt.grid()
# plt.xlabel("Frequency in Hertz",fontdict=font1)
# plt.ylabel("Magnitude response",fontdict=font1)
# plt.title("PSK signal spectrum v/s Frequency",fontdict=font2)
# plt.tick_params(axis = "both",labelsize = 16)  # ADJUSTING TICKS FONTSIZE
# plt.xlim(-40,40) # LIMITING AXIS
# plt.show()        