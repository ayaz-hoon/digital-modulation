# AUTHOR: AIYAZ AHMED N. HANSBHANVI
#( 201104003, TE-ETC Engineering, SEM-5,2021-22,GEC )
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft ,fftshift ,ifft ,ifftshift 
font1 = {'color':'black','size': 18} 
font2 = {'color':'black','size': 20}
fm = 3      # MESSAGE SIGNAL FREQUENCY
fs = 1000    # SAMPLING FREQUENCY
Vm = 2       # PEAK AMPLITUDE OF MESSAGE SIGNAL
dt = (1/fs)  # TIME PERIOD
t = np.arange(0,1,1/fs)  # TIME AXIS
vm = Vm*np.sin(2*np.pi*fm*t)  # MESSAGE SIGNAL EXPRESSION
L = len(t)    # LENGTH OF MESSAGE  SIGNAL
f = np.arange(-L/2,L/2,1)       # FREQUENCY AXIS
vm_shifted = vm+Vm     # DC SHIFTED SIGNAL 
plt.figure(1)   # MESSAGE SIGNAL V/S TIME PLOT
plt.subplot(3,1,1)
plt.plot(t,vm)
plt.xlabel("Time in Seconds" ,fontdict=font1)
plt.ylabel("Amplitude in Volts",fontdict=font1)
plt.title("Original Message Signal V/S Time ",fontdict=font1)
plt.tick_params(axis = "both",labelsize = 16)
#QUANTIZATION     
n = 4            # NO OF BITS
Vh = 4           # UPPER VOLTAGE
Vl = 0           # LOWER VOLTAGE
q = 2**n         # NO. OF INTERVALS 
s = (Vh - Vl)/q      # STEPSIZE   
# LOOP FOR QUANTIZATION
quan_list = []
for i in range (1,q+1):
    i *= s
    quan_list.append(i)
quan_sig = np.zeros(len(vm_shifted))
for i in range(len(vm_shifted)):
    for j in quan_list:
        if vm_shifted[i] == j:
            quan_sig[i] = j
            break
        if (vm_shifted[i]<j and vm_shifted[i]>= j-s):
            quan_sig[i] = j
                                                                                                                             
# ENCODING
# LOOP FOR ENCODING QUANTIZED SIGNAL
encode_list = []
for j in (quan_sig ):
    for i in range(len(quan_list)):
        if j == quan_list[i]:
           bin_data =  bin(i)
           encode_list.append(bin_data)
# DECODING
# LOOP FOR DECODING THE ENCODED DATA
decode_list = []
for j in encode_list:
   for i in range(len(quan_list)):
       if j == bin(i):   
            decoded_data = quan_list[i]
            decode_list.append(decoded_data)
# RECONSTRUCTION OF SIGNAL BY DECODED DATA
recons_sig =  np.zeros(len(quan_sig))
for i in range(len(quan_sig)):
    for j in range(len(decode_list)):
        if quan_sig[i] == decode_list[j] :
            recons_sig[i] = decode_list[j]       
# PLOT OF RECONSTRUCTED SIGNAL VS TIME
recons_sig_shift = recons_sig - Vm  # RECONSTRUCTED SIGNAL DC SHIFTED
plt.subplot(3,1,3)
plt.plot(t,recons_sig_shift)
plt.xlabel("Time in Seconds" ,fontdict=font1)
plt.ylabel("Amplitude in Volts",fontdict=font1)
plt.title("Reconstructed Message Signal V/S Time " ,fontdict=font1)
plt.tick_params(axis = "both",labelsize = 16)  
#SPECTRAL ANALYSIS
# FILTER DESIGN
bp_filtr = []
for x in f:
    if x > -(fm+5) and x < (fm +5):
        z = 1
        bp_filtr.append(z)
    else :
        z = 0
        bp_filtr.append(z)
    # TAKING FFT OF RECONSTRUCTED SIGNAL AND MULTIPLYING WITH BANDPASS FILTER
recons_spectrum = fftshift(fft(recons_sig_shift))*bp_filtr
recovd_sig = ifft(ifftshift(recons_spectrum))
# PLOTS OF ORIGINAL AND RECONSTRUCTED SIGNAL
plt.figure(2) 
plt.subplot(3,1,1)
plt.plot(t,vm)
plt.plot(t,recovd_sig , '--r')
plt.xlabel("Time in Seconds",fontdict=font1)
plt.ylabel("Amplitude in Volts",fontdict=font1)
plt.title("Original Message Signal and Recovered Signal V/S Time ",fontdict=font1)
plt.legend(["Original Signal" , "Recovered Signal"], loc = "upper right" )
plt.tick_params(axis = "both",labelsize = 16)   
plt.show()





