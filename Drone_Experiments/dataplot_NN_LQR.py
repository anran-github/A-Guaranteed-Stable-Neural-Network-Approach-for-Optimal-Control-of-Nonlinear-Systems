import scipy.io
import matplotlib.pyplot as plt
import numpy as np


# control number font size on axis.
# introduce latex
plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 16})
plt.rcParams["font.family"] = "Times New Roman"


# Load the .mat file
mat_LQR = scipy.io.loadmat('Drone_Experiments/drone_results/LQR1.MAT')
mat_NN = scipy.io.loadmat('Drone_Experiments/drone_results/YALMIP1.MAT')
mat_NOM = scipy.io.loadmat('Drone_Experiments/drone_results/NOM9.MAT')
# Access the data from the loaded .mat file
# Assuming the data is stored in a variable named 'data'
SS_LQR = mat_LQR['SS']
SS_NN = mat_NN['SS']
SS_NOM = mat_NOM['SS']
t_LQR = mat_LQR['Time'][1:,0] - mat_LQR['Time'][0,0]
t_NN = mat_NN['Time'][1:,0] - mat_NN['Time'][0,0]
t_NOM = mat_NOM['Time'][1:,0] - mat_NOM['Time'][0,0]

# Plot the data
# t = np.linspace()
start = 2222
start_NN = 965
start_NOM = 730
end = 3820
end_NN = 2050
end_NOM = 1800
# X direction
fig = plt.figure(figsize=(8, 8)) 
plt.subplot(311)
plt.plot(t_LQR[start:end]-t_LQR[start],SS_LQR[0,start:end],label='LQR')
plt.plot(t_NN[start_NN:end_NN]-t_NN[start_NN],SS_NN[0,start_NN:end_NN],label='YALMIP NN')
plt.plot(t_NOM[start_NOM:end_NOM]-t_NOM[start_NOM],SS_NOM[0,start_NOM:end_NOM],label='NOM NN')
plt.legend()
plt.xlim(0,14)
plt.ylabel('X Position [m]')
# plt.title('Drone Position Changes with Time.')
plt.grid(linestyle = '--')


# Y direction
plt.subplot(312)
plt.plot(t_LQR[start:end]-t_LQR[start],SS_LQR[1,start:end],label='LQR')
plt.plot(t_NN[start_NN:end_NN]-t_NN[start_NN],SS_NN[1,start_NN:end_NN],label='YALMIP NN')
plt.plot(t_NOM[start_NOM:end_NOM]-t_NOM[start_NOM],SS_NOM[1,start_NOM:end_NOM],label='NOM NN')
plt.ylabel('Y Position [m]')
plt.legend()
plt.xlim(0,14)
plt.grid(linestyle = '--')



# Z direction 
plt.subplot(313)
plt.plot(t_LQR[start:end]-t_LQR[start],SS_LQR[2,start:end],label='LQR')
plt.plot(t_NN[start_NN:end_NN]-t_NN[start_NN],SS_NN[2,start_NN:end_NN],label='YALMIP NN')
plt.plot(t_NOM[start_NOM:end_NOM]-t_NOM[start_NOM],SS_NOM[2,start_NOM:end_NOM],label='NOM NN')
plt.legend()
plt.xlim(0,14)
plt.xlabel('Time [s]')
plt.ylabel('Z Position [m]')
plt.grid(linestyle = '--')

plt.tight_layout()
plt.savefig('Drone_Experiments/Drone_NN_LQR.png',dpi=500)
plt.show()


# Analysis control effort.
control_lqr = mat_LQR['CI']
control_nn = mat_NN['CI']

for i in range(3):
    if i == 2:
        i += 1
    # print(i)
    # control_lqr = controls_lqr[i,:]
    # control_nn = controls_nn[i,:]
    lqr_ctrl_sum = np.sum(np.abs(control_lqr[i,start:end]))
    nn_ctrl_sum = np.sum(np.abs(control_nn[i,start:end]))

    plt.subplot(211)
    plt.plot(t_LQR[start:end]-t_LQR[start],control_lqr[i,start:end],label='LQR')
    _,x = plt.xlim()
    y,_ = plt.ylim()
    plt.annotate(f'Integrate Value:{lqr_ctrl_sum:.1f}',[x/3,y+0.01],bbox=dict(boxstyle="round", fc="none", ec="gray"))
    plt.legend()
    plt.ylabel('LQR Input')
    plt.grid()
    plt.title('Control Signal Changes with time')
    plt.subplot(212)
    plt.plot(t_LQR[start:end]-t_LQR[start],control_nn[i,start:end],label='NN')
    _,x = plt.xlim()
    y,_ = plt.ylim()
    plt.annotate(f'Integrate Value:{nn_ctrl_sum:.1f}',[x/3,y+0.01],bbox=dict(boxstyle="round", fc="none", ec="gray"))
    plt.grid()
    plt.ylabel('NN Input')
    plt.xlabel('Time [s]')
    plt.legend()
    plt.show()
    plt.close()
