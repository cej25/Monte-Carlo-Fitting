import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mc_chisq as mc
import time
plt.rcParams["font.family"] = "Times New Roman"
start_time = time.time()


#number of random samples. 10^4+ recommended
trials = 10000
#change name of csv formatted file here for input data
filename = '99Y900msradius'
data = pd.read_csv(filename+'.csv')

#must have headers "Bin" and "Frequency"
bins = data["Bin"]
freq = data["Frequency"]
x = []
y = []
yerr = []

#adjust range here with rows that need fitting: (lower, upper+1) 16,58 original
for i in range(12,48):
    x.append(bins.loc[i])
    y.append(freq.loc[i])
    yerr.append(freq.loc[i]**0.5)
    z = bins.loc[i+1]

xx = np.linspace(x[0],x[len(x)-1],1000)


#options "linear", "exponential", "gaussian", "double gaussian"
function = "gaussian"

#adjust for double gaussian where sigma is fixed
sig1 = 18.60275
sig2 = 18.60275

#four parameters needed for double gauss, three for single gauss, etc.
p0 = [30,7,2]

shape = (function, sig1, sig2, p0)

fit_data = mc.fit(x, y, yerr, trials, shape)

xlims = [0,0]
ylims = [0,0]

hist_face_color = 'b'
hist_edge_color = 'w'
fit_line = 'm-'
mean_line = 'r-'
std_fill_color = 'r'

# plotted for skewed distributions 
median_line = 'g-'
iqr_fill_color = 'g'

design = (xlims,ylims,hist_face_color, hist_edge_color, fit_line, mean_line, median_line,
          std_fill_color, iqr_fill_color)


info = {
        'xlabel': '',
        'ylabel': 'Probability',
        'savefig': ''
        }

for i in range(len(fit_data)):
    
    if (function == "double gaussian"):
        if (i == 0):
            info['xlabel'] = r'$A_{\alpha}$'
            info['savefig'] = filename + '_double_hist_A_alpha.pdf'
            print("A_alpha: %.2f ± %.2f" %(fit_data[i][4],fit_data[i][5]))
            min_chisq = 0
        elif (i == 1):
            info['xlabel'] = r'$\mu_{\alpha}$'
            info['savefig'] = filename + '_double_hist_mu_alpha.pdf'
            print("mu_alpha: %.2f ± %.2f" %(fit_data[i][4],fit_data[i][5]))
            min_chisq = 0
        elif (i == 2):
            info['xlabel'] = r'$A_{\beta}$'
            info['savefig'] = filename + '_double_hist_A_beta.pdf'
            print("A_beta: %.2f ± %.2f" %(fit_data[i][4],fit_data[i][5]))
            min_chisq = 0
        elif (i == 3):
            info['xlabel'] = r'$\mu_{\beta}$'
            info['savefig'] = filename + '_double_hist_mu_beta.pdf'
            print("mu_beta: %.2f ± %.2f" %(fit_data[i][4],fit_data[i][5]))
            min_chisq = 0
        elif (i == 4):
            info['xlabel'] = r'$\mu_{diff}$'
            info['savefig'] = filename + '_double_hist_mu_diff.pdf'
            print("mu_diff: %.2f ± %.2f" %(fit_data[i][4],fit_data[i][5]))
            min_chisq = 0
        elif (i == 5):
            info['xlabel'] = r'Production ratio of ${\alpha}$'
            info['savefig'] = filename + '_double_hist_pr_alpha.pdf'
            print("pr_alpha: %.2f ± %.2f" %(fit_data[i][4],fit_data[i][5]))
            min_chisq = 0.2706
        elif (i == 6):
            info['xlabel'] = r'Production ratio of ${\beta}$'
            info['savefig'] = filename + '_double_hist_pr_beta.pdf'
            print("pr_beta: %.2f ± %.2f" %(fit_data[i][4],fit_data[i][5]))
            min_chisq = 0.7294
    
    elif (function == "gaussian"):
        if (i == 0):
            info['xlabel'] = 'A'
            info['savefig'] = filename + 'single_hist_A.pdf'
            print("A: %.2f ± %.2f" %(fit_data[i][4],fit_data[i][5]))
            min_chisq = 0
        elif (i == 1):
            info['xlabel'] = r'$\mu$'
            info['savefig'] = filename + 'single_hist_mu.pdf'
            print("mu: %.2f ± %.2f" %(fit_data[i][4],fit_data[i][5]))
            min_chisq = 0
        elif (i == 2):
            info['xlabel'] = r'$\sigma$'
            info['savefig'] = filename + 'single_hist_sigma.pdf'
            print("sigma: %.2f ± %.2f" %(fit_data[i][4],fit_data[i][5]))
            min_chisq = 0
            

    mc.plotHist(fit_data[i], design, info, min_chisq)




### ==== actual data plot-design ==== ###

plot_data = [x,y,z]
for i in range(len(fit_data)):
    plot_data.append(fit_data[i][4])

hist_face_color = 'b'
hist_edge_color = 'w'
fit_line = 'r-'

# needed for double gaussian
fit_alpha_line = 'g--'
fit_beta_line = 'b--'

#xlims = [40,180]
xlims = [0,0]
ylims = [0,0]

design = (xlims,ylims,hist_face_color,hist_edge_color,fit_line,fit_alpha_line,fit_beta_line)

info['xlabel'] = r'Phase angle ($\phi$)'
info['ylabel'] = 'Counts'
info['savefig'] = filename + '_double.pdf'


mc.plotData(plot_data,shape,design,info)



### ==== chi-squared calculation ==== ###

params = []
if (function == 'double gaussian'):
    n = 4
elif (function == 'gaussian'):
    n = 3
for i in range(n):
        params.append(fit_data[i][4])

chisq, chisq_red = mc.calc_chisq(shape, x, y, params)
print("Chi-Squared: %.2f" %(chisq))
print("Reduced Chi-Squared: %.2f" %(chisq_red))



print("\nRun time: %.2f"%(time.time() - start_time), " seconds")