#visualize a cure cycle given a 12 dimensional array
import matplotlib.pyplot as plt
import numpy as np

def plot_inputs(ti, vi, ai, room_temp = 23, ap_initial = 100, vp_initial = 100): 
    '''
    plots temperature and pressure separately.
    '''
    plt.figure(figsize = (15,5))

    plt.subplot(1, 2,1)
    x1 = np.linspace(0, ti[1])
    y1 = room_temp + x1*ti[0]
    plt.plot(x1, y1, 'k')

    x2 = np.linspace(ti[1], ti[1]+ti[2])
    y2 = x2*0 + y1[-1]
    plt.plot(x2, y2, 'k')

    x3 = np.linspace(ti[1]+ti[2], ti[1]+ti[2]+ti[4])
    x3_ = np.linspace(0, ti[4])
    y3 = y1[-1] + x3_*ti[3]
    plt.plot(x3, y3, 'k')

    x4 = np.linspace(ti[1]+ti[2]+ti[4], ti[1]+ti[2]+ti[4]+ti[5])
    y4 = y3[-1] + x4*0
    plt.plot(x4, y4, 'k', label = 'Autoclave temperature')

    plt.legend()
    plt.grid()

    #vaccume plot 
    plt.subplot(1,2,2)
    xv1 = np.linspace(0, vi[1])
    yvi = vp_initial + xv1*0

    yvi[-1] = vi[0]*100
    plt.plot(xv1, yvi, '-b')

    xv2 = np.linspace(vi[1], vi[1]+vi[2])
    yv2 = vi[0]*100 + xv2*0
    plt.plot(xv2, yv2, '-b', label = 'vaccume pressure')
    plt.legend()

    #autoclave pressure 
    xp1 = np.linspace(0, ai[1])
    ypi = ap_initial + xp1*0

    ypi[-1] = ai[0]*100
    plt.plot(xp1, ypi, '-r')

    xa2 = np.linspace(ai[1], ai[1]+ai[2])
    ya2 = ai[0]*100 + xa2*0
    plt.plot(xa2, ya2, '-r', label = 'autoclave pressure')
    plt.legend()
    plt.grid()
    plt.ylabel('K Pa')
    #plt.xlim([0, 250])
    plt.show()

    print(f"t1-{ti[1]+ti[2]+ti[4]+ti[5]}, t2-{vi[1]+vi[2]}, t3-{ai[1]+ai[2]}")
    return 1 

def plot_inputs_v2(ti, vi, ai, room_temp = 23, ap_initial = 100, vp_initial = 100): 
    
    '''
    plots temperature and plots in the same plot.
    
    '''
    
    from matplotlib import rc
    rc('mathtext', default='regular')
    
    plt.figure(figsize = (15,5))
    
    
    ig, ax1 = plt.subplots(figsize = (10,5))

    ax2 = ax1.twinx()
    #ax1.plot(x, y1, 'g-')
    #ax2.plot(x, y2, 'b-')

    ax1.set_xlabel('X data')
    ax1.set_ylabel('Temperature (K)', color='g')
    ax2.set_ylabel('Pressure (K pa)', color='c')

    
    #plt.subplot(1, 2,1)
    x1 = np.linspace(0, ti[1])
    y1 = room_temp + x1*ti[0]
    ax1.plot(x1, y1, 'k')

    x2 = np.linspace(ti[1], ti[1]+ti[2])
    y2 = x2*0 + y1[-1]
    ax1.plot(x2, y2, 'k')

    x3 = np.linspace(ti[1]+ti[2], ti[1]+ti[2]+ti[4])
    x3_ = np.linspace(0, ti[4])
    y3 = y1[-1] + x3_*ti[3]
    ax1.plot(x3, y3, 'k')

    x4 = np.linspace(ti[1]+ti[2]+ti[4], ti[1]+ti[2]+ti[4]+ti[5])
    y4 = y3[-1] + x4*0
    ax1.plot(x4, y4, 'k', label = 'Autoclave temperature')

    plt.grid()
    
    #vaccume plot 
    #plt.subplot(1,2,2)
    xv1 = np.linspace(0, vi[1])
    yvi = vp_initial + xv1*0

    yvi[-1] = vi[0]*100
    ax2.plot(xv1, yvi, '-b')

    xv2 = np.linspace(vi[1], vi[1]+vi[2])
    yv2 = vi[0]*100 + xv2*0
    ax2.plot(xv2, yv2, '-b', label = 'vaccume pressure')
    
    
    #autoclave pressure 
    xp1 = np.linspace(0, ai[1])
    ypi = ap_initial + xp1*0

    ypi[-1] = ai[0]*100
    ax2.plot(xp1, ypi, '-r')
    ax1.grid()
    xa2 = np.linspace(ai[1], ai[1]+ai[2])
    ya2 = ai[0]*100 + xa2*0
    ax2.plot(xa2, ya2, '-r', label = 'autoclave pressure')
    plt.grid()
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc=5)

    #ax2.ylabel('K Pa')
    #plt.xlim([0, 250])
    plt.show()

    print(f"t1-{ti[1]+ti[2]+ti[4]+ti[5]}, t2-{vi[1]+vi[2]}, t3-{ai[1]+ai[2]}")
    return 1 


def push_to_discrete(x_sample, unique_vals):
    '''
    for each value in x (xi)
    push it to the closest available value in unique_vals
    return pushed input 
    '''
    discrete_x = []
    for i, xi in enumerate(x_sample):
            xi_available_values = unique_vals[i]
            differences = (xi_available_values - xi)**2
            min_index = np.argmin(differences)
            discrete_x.append(xi_available_values[min_index])
    return discrete_x


def plot_outputs(y_arr, y_mins, y_maxs, labels = []):
    
    plt.figure(figsize = (20, 7))
    
    for i in range(len(y_arr)): 

        plt.subplot(1,7,i+1)

        label = labels[i]

        y_min = y_mins[i]
        y_max = y_maxs[i]
        y_curr = y_arr[i]
        
        plt.plot([y_min],'-og', label='Min',  markersize=25)
        plt.plot([y_max], '-or', label='Max',  markersize=25)
        plt.plot([y_curr], '-ob', label  = 'Opt',  markersize=20)

        #ax.set_ylabel('Scores')
        #ax.set_title('Scores by group and gender')
        #ax.legend()
        plt.title(label)
        
    plt.legend(loc = 4)
    plt.show()
    
    return 1


def load_general_model(verbose = False):
    '''
    a model was trained on 12 inputs and 7 outputs
    this model will be returned based on saved weights
    this is a dense resnet 
    
    Verbose: enables model.summary()
    '''
    
    from keras.layers import Add, Convolution2D, Input, Dense, Dropout
    from tensorflow.keras import regularizers
    import tensorflow as tf

    hlen = 256
    dout = 0.1

    x = Input(shape=(12,))
    XH1_ = Dense(hlen,
                 activation = 'relu',
                 use_bias = True,
                 kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(x)
    XH1  = Dropout(dout)(XH1_)
    XL1 = Dense(hlen, activation = 'linear')(x)

    x1 = Add()([XL1,XH1])

    XH2_ = Dense(hlen,
                 activation = 'relu',
                 use_bias = True,
                 kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(x1)
    XH2  = Dropout(dout)(XH2_)
    XL2 = Dense(hlen, activation = 'linear')(x1)
    x2 = Add()([XL2,XH2, x1])

    XH3_ = Dense(hlen,
                 activation = 'relu',
                 use_bias = True,
                 kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(x2)
    XH3  = Dropout(dout)(XH3_)
    XL3 = Dense(hlen, activation = 'linear')(x2)
    x3 = Add()([XL3,XH3, x1, x2])

    XH4_ = Dense(hlen,
                 activation = 'relu',
                 use_bias = True,
                 kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(x3)
    XH4  = Dropout(dout)(XH4_)
    XL4 = Dense(hlen, activation = 'linear')(x3)
    x4 = Add()([XL4,XH4, x1, x2, x3])

    XH5_ = Dense(hlen,
                 activation = 'relu',
                 use_bias = True,
                 kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(x4)
    XH5  = Dropout(dout)(XH5_)
    XL5 = Dense(hlen, activation = 'linear')(x4)
    x5 = Add()([XL4,XH4, x1, x2, x3, x4])


    xout = Dense(2)(x5)

    model  = tf.keras.Model(inputs = x, outputs = xout, name = 'ACS_ResNet')

    if verbose: model.summary()

    path  = "best_model_resnet_v2"

    tf.random.set_seed(123)

    def cus_loss(y_true, y_pred):

        return tf.norm(y_true-y_pred, ord = 2)/tf.norm(y_pred, ord = 2)

    model.compile(optimizer = 'adam', loss = cus_loss, metrics = ['mse'])

    return model 

if __name__ == "__main__" : 
    
    #using the plot_inputs() function 
    
    case = -1
    ti = temp_inputs[case]
    ai = autoclave_inputs[case]
    vi = vaccum_inputs[case]


    plot_inputs(ti, vi, ai)
