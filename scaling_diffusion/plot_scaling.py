#time to run 10k iterations (Lazy mode)

import matplotlib.pyplot as plt
import matplotlib.ticker

plt.rcParams.update({
    'font.size': 20,           # Increase default font size
    'axes.labelsize': 22,      # Axis label font size
    'axes.titlesize': 24,      # Title font size
    'legend.fontsize': 18,     # Legend font size
    'xtick.labelsize': 18,     # X-tick label size
    'ytick.labelsize': 18      # Y-tick label size
})

#Gaudi1
time_voy_g1=[1440,764,385,195,100,52,24]
hpus_voy_g1=[1,2,4,8,16,32,64]

#Gaudi2 
time_voy_g2=[350,195,100,53]
hpus_voy_g2=[1,2,4,8]

#Gaudi3 
time_voy_g3=[276,144,80,44]
hpus_voy_g3=[1,2,4,8]

#expanse (V100)
time_exp=[750,393,193]
gpus_exp=[1,2,4]

#expanse a100
time_exp_a100=[305,164,90]
gpus_exp_a100=[1,2,4]

#expanse h100
time_exp_h100=[132,72,43]
gpus_exp_h100=[1,2,4]

fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [2, 1]}
)

desired_x_ticks = [1, 2, 4, 8, 16, 32, 64]
desired_y_ticks = [10, 30, 100, 300, 1000]

fig.suptitle('Training Super-Resolution: 40k images', fontsize=22)

# First plot (linear one)
ax1.plot(hpus_voy_g1,time_voy_g1,marker='s',label='Gaudi',color='g')
ax1.plot(hpus_voy_g2,time_voy_g2,linestyle='-',marker='s',label='Gaudi2',color='b')
ax1.plot(hpus_voy_g3,time_voy_g3,linestyle='-',marker='s',label='Gaudi3',color='m')
ax1.plot(gpus_exp,time_exp,linestyle=':',marker='d',label='V100',color='g')
ax1.plot(gpus_exp_a100,time_exp_a100,linestyle=':',marker='d',label='A100',color='b')
ax1.plot(gpus_exp_h100,time_exp_h100,linestyle=':',marker='d',label='H100',color='m')
ax1.set_xlim(1, 8)
ax1.set_ylim(0, 1000)
ax1.set_xlabel('Number of devices')
ax1.set_ylabel('Time(min)')
ax1.legend()

# Second plot (log-log one)
ax2.plot(hpus_voy_g1,time_voy_g1,marker='s',label='Gaudi',color='g')
ax2.plot(hpus_voy_g2,time_voy_g2,linestyle='-',marker='s',label='Gaudi2',color='b')
ax2.plot(hpus_voy_g3,time_voy_g3,linestyle='-',marker='s',label='Gaudi3',color='m')
ax2.plot(gpus_exp,time_exp,linestyle=':',marker='d',label='V100',color='g')
ax2.plot(gpus_exp_a100,time_exp_a100,linestyle=':',marker='d',label='A100',color='b')
ax2.plot(gpus_exp_h100,time_exp_h100,linestyle=':',marker='d',label='H100',color='m')
ax2.set_yscale('log')
ax2.set_xscale('log')
ax2.set_xticks(desired_x_ticks)
ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax2.set_yticks(desired_y_ticks)
ax2.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax2.set_ylabel('Time(min)  - Log scale')
ax2.set_xlabel('Number of devices - Log scale')

#ax2.legend()

plt.tight_layout()
plt.savefig('scaling_diffusion.png')
plt.show()