# AUTHOR: AIYAZ AHMED N.HANSBHANVI
#( 201104003, TE-ETC Engineering, SEM-5,2021-22,GEC )
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft ,fftshift 
font1 ={"color":"black", "size":18}
font2 ={"color":"black", "size":20}
bit_seq = [0,1,1,0,1,0,1,0,1,0,1,0,1]     # BIT SEQUENCE
num_bits = len(bit_seq)   # NUMBER OF BITS
fs = 1000                  # SAMPLING FREQUENCY
fc = 20                   # CARRIER FREQUENCY
t = np.arange(0, num_bits,1/fs)  # TIME INDEX
L =  len(t)               # LENGTH OF TIME AXIS
t_end = L/fs
f =np.arange(-L/2,L/2,1)*1/t_end  # FREQUENCY AXIS
t_size = len(t)
Vc =2                     # PEAK AMPLITUDE OF  CARRIER SIGNAL
vc = Vc*np.sin(2*np.pi*fc*t)    # CARRIER SIGNAL
#plt.plot(t,vc)
#plt.show()
Tb = t_end/num_bits       # BIT DURATION
# CREATING LIST OF ZEROS AND ONES OF BIT DURATION Tb
zeros = []
for i in range(1,t_size):
    if t[i] <= Tb:
        z = 0
        zeros.append(z)
        
ones = []
for j in range(1,t_size):
    if t[j]<=Tb:
        o = 1
        ones.append(o)

# CREATING SQUARE WAVE ACCORDING TO THE BIT SEQUENCE
sqr_wave = []
for j in range(len(bit_seq)):
    if bit_seq[j] == 0:
        sqr_wave = sqr_wave+zeros
        
    else:
        sqr_wave = sqr_wave+ones

Vask = sqr_wave*vc  # EXPRESSION FOR ASK SIGNAL
Vask_spectrum = fftshift(fft(Vask))  # FFT OF ASK SIGNAL  

# PLOTS IN TIME DOMAIN          
plt.figure(1)
plt.subplot(3,1,1)       
plt.plot(t,sqr_wave) # MODULATING SIGNAL V/S TIME 
plt.grid()
plt.xlabel("Time in seconds",fontdict=font1)
plt.ylabel("Amplitude in volts",fontdict=font1)
plt.title("Modulating Signal v/s Time",fontdict=font2)
plt.legend([bit_seq],loc = "lower right")
plt.tick_params(axis = "both",labelsize = 16)

plt.subplot(3,1,3)
plt.plot(t,vc) # CARRIER SIGNAL V/S TIME 
plt.grid()
plt.xlabel("Time in seconds",fontdict=font1)
plt.ylabel("Amplitude in volts",fontdict=font1)
plt.title("Carrier Signal v/s Time",fontdict=font2)
plt.tick_params(axis = "both",labelsize = 16)

plt.figure(2)
plt.subplot(3,1,1)
plt.plot(t,Vask)
plt.grid()
plt.xlabel("Time in seconds",fontdict=font1)
plt.ylabel("Amplitude in volts",fontdict=font1)
plt.title("ASK Signal v/s Time",fontdict=font2)
plt.tick_params(axis = "both",labelsize = 16)

#FFREQUENCY SPECTRUM
plt.subplot(3,1,3)
plt.plot(f,abs(Vask_spectrum))
plt.grid()
plt.xlim(-60,60)
plt.tick_params(axis = "x",labelsize = 16)
plt.xlabel("Frequency in Hertz",fontdict=font1)
plt.ylabel("Magnitude Response",fontdict=font1)
plt.title("ASK Signal Spectrum v/s Hertz",fontdict=font2)
plt.show()    
      
        
               