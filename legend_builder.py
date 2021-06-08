import pylab
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

# line1 = mlines.Line2D([],[],color='black',linestyle='dashed',label=r'$N=51$')
line2 = mlines.Line2D([],[],color='teal',linestyle='dashed',label=r'$N=801$')
# line3 = mlines.Line2D([],[],color='teal',linestyle='dashed',label=r'$N=201$')
# line4 = mlines.Line2D([],[],color='navy',linestyle='dashed',label=r'$N=401$')
# line5 = mlines.Line2D([],[],color='forestgreen',linestyle='dashed',label=r'$N=801$')
# line6 = mlines.Line2D([],[],color='red',linestyle='dashed',label=r'$N=1601$')
# line7 = mlines.Line2D([],[],color='silver',linestyle='dashed',label=r'$r=8$')
figlegend = pylab.figure(figsize=(2,0.2))
figlegend.legend(handles=[line2],loc='center',frameon=False)
# .legend(frameon=False)
figlegend.savefig('801_grid_legend.png')
# figlegend.show()
# fig = pylab.figure()
# figlegend = pylab.figure(figsize=(3,3))
# ax = fig.add_subplot(111)
# x = np.linspace(0, 2 * np.pi, 400)
# line1 = ax.plot([1,2,3])
# line2 = ax.plot([1,2,3])
# # lines = ax.plot(range(10), pylab.randn(10), range(10), pylab.randn(10))
# figlegend.legend(handles=[line1,line2], (r'$N=21$', r'$N=41$', r'$N=81$', r'$N=161$', r'$N=321$', r'$N=641$'), 'center')
# fig.show()
# # figlegend.show()
# figlegend.savefig('legend.png')