# AUTHOR: AIYAZ AHMED N.HANSBHANVI
#( 201104003, TE-ETC Engineering, SEM-5,2021-22,GEC )
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft ,fftshift 
font1 ={"color":"black", "size":18}
font2 ={"color":"black", "size":20}   # FOR FONT SIZE
bit_seq = [0,1,0,1,0]     # BIT SEQUENCE
num_bits = len(bit_seq)   # NUMBER OF BITS
fs = 500                  # SAMPLING FREQUENCY
fc = 20                   # CARRIER FREQUENCY
t = np.arange(0, 5,1/fs)  # TIME INDEX
L =  len(t)               # LENGTH OF TIME AXIS
f =np.arange(-L/2,L/2,1)*1/5  # FREQUENCY AXIS
fo = 10
Vc =2                     # PEAK AMPLITUDE OF  CARRIER SIGNAL
vc = Vc*np.sin(2*np.pi*fc*t)    # CARRIER SIGNAL
fh = fc + fo    # HIGHER FREQUENCY FOR GENERATION OF FSK
fl = fc - fo    # LOWER FREQUENCY FOR GENERATION OF FSK
Tb = 5/num_bits       # BIT DURATION
# CREATING LIST OF ZEROS AND ONES OF BIT DURATION Tb
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

# CREATING BIPOLAR SQUARE WAVE ACCORDING TO THE BIT SEQUENCE
d_h = []
for j in range(len(bit_seq)):
    if bit_seq[j] == 0:
        d_h = d_h + neg_ones
        
    else:
        d_h = d_h + ones

#LIST CONTAINING 1 FOR LOWER VOLTAGE VALUE AND ZERO FOR HIGHER VOLTAGE VALUE
p_l = []
for i in range(len(d_h)):
    if d_h[i] < 0:
        va_l = 1
        p_l.append(va_l)
        
    else:
        va_l = 0
        p_l.append(va_l)

#LIST CONTAINING 1 FOR HIGHER VOLTAGE VALUE AND ZERO FOR LOWER VOLTAGE VALUE
p_h = []
for i in range(len(d_h)):
    if d_h[i] > 0:
        va_l2 = 1
        p_h.append(va_l2)
        
    else:
        va_l2 =0
        p_h.append(va_l2)
# GENERATION OF FSK SIGNAL          
v1 = Vc*np.sin(2*np.pi*fh*t)*p_h
v2 = Vc*np.sin(2*np.pi*fl*t)*p_l
vfsk = v1 + v2       # FSK SIGNAL
v_spec = fftshift(fft(vfsk))     # TAKING FFT OF FSK SIGNAL
# TOME DOMAIN PLOTS
plt.figure(1)
plt.subplot(3,1,1)
plt.plot(t,d_h)
plt.grid()
plt.xlabel("Time in seconds",fontdict=font1)
plt.ylabel("Amplitude in volts",fontdict=font1)
plt.title("Modulating Signal v/s Time",fontdict=font2)
plt.legend(["Bits =(1,1,0,1,0)"],loc = "lower right")
plt.tick_params(axis = "both",labelsize = 16)  # ADJUSTING TICKS FONTSIZE

plt.subplot(3,1,3)
plt.plot(t,vc)
plt.grid()
plt.xlabel("Time in seconds",fontdict=font1)
plt.ylabel("Amplitude in volts",fontdict=font1)
plt.title("Carrier Signal v/s Time",fontdict=font2)
plt.tick_params(axis = "both",labelsize = 16)    # ADJUSTING TICKS FONTSIZE

plt.figure(2)
plt.subplot(3,1,1)
plt.plot(t,vfsk)
plt.grid()
plt.xlabel("Time in seconds",fontdict=font1)
plt.ylabel("Amplitude in volts",fontdict=font1)
plt.title("FSK Signal v/s Time",fontdict=font2)
plt.tick_params(axis = "both",labelsize = 16)   # ADJUSTING TICKS FONTSIZE

#FREQUENCY DOMAIN PLOT
plt.subplot(3,1,3)

# plt.subplot(2,1,2)
plt.plot(f,abs(v_spec))
plt.grid()
plt.xlabel("Frequency in Hertz",fontdict=font1)
plt.ylabel("Magnitude response",fontdict=font1)
plt.title("FSK Signal Spectrum v/s Frequency",fontdict=font2)
plt.tick_params(axis = "both",labelsize = 16)    # ADJUSTING TICKS FONTSIZE
plt.xlim(-100,100)
plt.show()
    