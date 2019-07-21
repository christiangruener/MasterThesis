import logging, os, sys, inspect, time
import matplotlib.pyplot as plt
import matplotlib.font_manager
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
import tensorflow as tf
from tensorflow.python.training import moving_averages

import myproblems

TF_DTYPE = tf.float64

### Default parameters
default_options = {}
## batch normalization values:
default_options['apply_batch_normalization'] = True
default_options['epsilon'] = 1e-8
default_options['initial_beta_std'] = 1.
default_options['initial_gamma_minimum'] = 0.
default_options['initial_gamma_maximum'] = 1.
default_options['initial_gamma_minimum'] = .1
default_options['initial_gamma_maximum'] = .5
default_options['momentum'] = 0.99
## PDE initialization values
default_options['initial_gradient_minimum'] = -.1
default_options['initial_gradient_maximum'] =.1
default_options['initial_value_minimum'] = 1
default_options['initial_value_maximum'] = 5
## Values for algorithm
default_options['learning_rates'] = [1e-2]
default_options['learning_rate_boundaries'] = []
default_options['number_of_iterations'] = 4000
default_options['number_of_hidden_layers'] = 4
default_options['number_of_neurons_per_hidden_layer'] = None # default: dim + 10, but dim accessible yet
default_options['number_of_test_samples'] = 256
default_options['number_of_time_intervals'] = 20
default_options['number_of_training_samples'] = 256
## for backtesting
default_options['logging_frequency'] = 25
default_options['testing_samplerange'] = 5 #no of indepdent runs
default_options['produce_summary'] = True


def solve(currentProblem, session=tf.Session(), **vars ):
    ## Get the default options
    options = default_options.copy()
    options.update(vars)
    # create storage room for all additional training operations
    additional_training_ops = []
    # is the model getting trained?
    is_getting_trained = tf.placeholder(tf.bool)
    ## initialize layers and batch normalization
    def nn_layers(inputs, number_of_neurons, activation = None):
        # For introducing weights use xavier_initializer
        xavier_initializer = tf.contrib.layers.xavier_initializer()
        if options['apply_batch_normalization']:
            biases_initializer = tf.zeros_initializer()
        else:
            biases_initializer = xavier_initializer
            
        result = tf.contrib.layers.fully_connected(
            inputs, 
            number_of_neurons,
            activation_fn = None,
            weights_initializer=xavier_initializer,
            biases_initializer=biases_initializer
        )

        if options['apply_batch_normalization']:
            # shape_of_paras = [result.get_shape()[-1]]
            shape_of_paras = result.get_shape()
            beta = tf.get_variable(
                'beta',
                shape = [shape_of_paras[-1]],
                dtype = TF_DTYPE,
                initializer = tf.random_normal_initializer(
                    mean = 0.,
                    stddev = options['initial_beta_std'],
                    dtype = TF_DTYPE
                )
            )
            gamma = tf.get_variable(
                'gamma',
                shape = [shape_of_paras[-1]],
                dtype = TF_DTYPE,
                initializer = tf.random_uniform_initializer(
                    minval = options['initial_gamma_minimum'],
                    maxval = options['initial_gamma_maximum'],
                    dtype = TF_DTYPE)
            )

            moving_mean = tf.get_variable(
                'moving_mean',
                shape = [shape_of_paras[-1]],
                dtype = TF_DTYPE,
                initializer = tf.constant_initializer(0.0, TF_DTYPE),
                trainable = False
            )
            moving_variance = tf.get_variable(
                'moving_variance',
                shape = [shape_of_paras[-1]],
                dtype = TF_DTYPE,
                initializer = tf.constant_initializer(1.0, TF_DTYPE),
                trainable = False
            )

            mean, variance = tf.nn.moments(result, [0])
            additional_training_ops.append(
                moving_averages.assign_moving_average(
                    moving_mean,
                    mean,
                    options['momentum'] #decay
                )
            )
            additional_training_ops.append(
                moving_averages.assign_moving_average(
                    moving_variance,
                    variance,
                    options['momentum'] #decay
                )
            )
            mean, variance = tf.cond(
                is_getting_trained,
                lambda: (mean, variance), #if is_getting_trained == true
                lambda: (moving_mean, moving_variance) # if is_getting_trained != true
            )
            result = tf.nn.batch_normalization(
                result,
                mean, variance,
                beta, gamma,
                options['epsilon']
            )
            result.set_shape(shape_of_paras)
        if activation:
            return activation(result)
        return result #no activiation for output layer

    ### Setup Building Structure of Neuronal Network

    # steps in time and create partition of time steps t_0 < t_1 < ... < t_{N-1}
    dt = currentProblem.terminal_time / options['number_of_time_intervals']
    time_partition = np.arange(0, options['number_of_time_intervals']) * dt
    # Create a Placeholder for the Brownian Motion
    dW = tf.placeholder(
        dtype=TF_DTYPE,
        shape=[
            None, #because no. of test samples could differ from no. of training samples
            currentProblem.dimensions,
            options['number_of_time_intervals']
        ]
    )
    # Create a Placeholder for the Process
    X = tf.placeholder(
        dtype=TF_DTYPE,
        shape=[
            None,
            currentProblem.dimensions,
            options['number_of_time_intervals'] + 1
        ]
    )
    # Initalize value at time zero with wildly guessing: between 0. and 1.
    Y_init = tf.get_variable(
        'Y_init',
        shape=[], #scalar
        dtype=TF_DTYPE,
        initializer=tf.random_uniform_initializer(
            minval=options['initial_value_minimum'],
            maxval=options['initial_value_maximum'],
            dtype=TF_DTYPE
        )
    )
    # Initalize gradient at time zero between 0. and 1.
    Z_init = tf.get_variable(
        'Z_init',
        shape=[1, currentProblem.dimensions],
        dtype=TF_DTYPE,
        initializer=tf.random_uniform_initializer(
            minval=options['initial_gradient_minimum'],
            maxval=options['initial_gradient_maximum'],
            dtype=TF_DTYPE
        )
    )
    # for later usage create a vector with all ones
    ones = tf.ones(
        shape=tf.stack( [tf.shape(dW)[0], 1] ), # just for right dim use dW
        dtype=TF_DTYPE
    )
    if options['produce_summary']:
        tf.summary.scalar('Y_0', Y_init)
    # Number of neurons per hidden layer
    number_of_neurons = options['number_of_neurons_per_hidden_layer']
    if number_of_neurons is None:  # dimension is set inside problem, therefore no default value
        number_of_neurons = currentProblem.dimensions + 10

    ##
    ## Building Neuronal Network...
    ##
    logging.debug('Building neural network...')
    # Simulate process from starting to terminal time
    Y = ones * Y_init
    Z = tf.matmul(ones, Z_init)
    i = 0
    while True: # Y_{t_{n+1}} ~= Y_{t_n} - f dt + Z_{t_n} dW
        Y = Y - currentProblem.generator(time_partition[i], X[:,:, i], Y, Z) * dt \
            + tf.reduce_sum(Z * dW[:,:,i], axis = 1, keepdims= True)
        if options['produce_summary']:
            tf.summary.scalar('E_Y_{}'.format(i + 1), tf.reduce_mean(Y))
        i = i + 1
        if i == options['number_of_time_intervals']:
            # No need to approximate Z_T, so break here
            break
        with tf.variable_scope('t_{}'.format(i), reuse= tf.AUTO_REUSE ):
            tmp = X[:,:,i]
            # Create hidden layers:
            for k in range(1, options['number_of_hidden_layers'] + 1):
                with tf.variable_scope('layer_{}'.format(k), reuse= tf.AUTO_REUSE ):
                    tmp = nn_layers(
                        tmp,
                        number_of_neurons,
                        tf.nn.relu #activation fnct Re(ctified) L(inear) (U)nit
                    )
            # Create output layer #-> no activation fct
            tmp = nn_layers(
                tmp,
                currentProblem.dimensions
            )
            Z = tmp / currentProblem.dimensions
    # Create loss function as least mean squared error
    loss = \
        tf.reduce_mean(
        tf.square(
            Y - currentProblem.terminal_condition(
            X[:, :, -1])
        )
    )
    if options['produce_summary']:
        tf.summary.scalar('loss', loss)
    # Training Operations
    global_step = tf.get_variable(
        'global_step',
        shape=[], #scalar
        dtype= tf.int32,
        trainable=False,
        initializer=tf.constant_initializer(0)
    )
    # Initialize learning rate if boundaries exist
    learning_rate = options['learning_rates']
    learning_rate_boundaries = options['learning_rate_boundaries']
    assert(len(learning_rate) == len(learning_rate_boundaries)+1) #size have to fit
    if len(learning_rate) == 1:
        learning_rate_function = learning_rate[0]
    else:
        learning_rate_function = tf.train.piecewise_constant(
            global_step,
            learning_rate_boundaries,
            learning_rate
        )
    # get trainable obs and apply gradient to it
    trainable_variables = tf.trainable_variables()
    gradients = tf.gradients(loss,trainable_variables)
    optimizer = tf.train.AdamOptimizer(learning_rate_function)
    gradient_update = optimizer.apply_gradients(
        zip(gradients,
            trainable_variables
        ), #combine gradients to related trainable_variables
        global_step
    )
    tmp = [gradient_update] + additional_training_ops
    training_operations = tf.group(*tmp)
    if options['produce_summary']:
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(
            FLAGS.summaries_directory,
            session.graph
        )

    ##
    ## Start Training
    ##
    logging.debug('Network Training... ')

    # Create Test samples:
    dW_test, X_test = currentProblem.sample(
        options['number_of_test_samples'],
        options['number_of_time_intervals']
    )
    test_feed_dictionary = {
        dW: dW_test,
        X: X_test,
        is_getting_trained: False
    }
    # Create veariables for  for Backtesting
    save_Y_init = np.empty([options['number_of_iterations'] + 1])
    save_loss = np.empty([options['number_of_iterations'] + 1])
    session.run(tf.global_variables_initializer())
    for steps in range(options['number_of_iterations'] +1):
        #just logging:
        observed_loss, observed_Y_init = session.run(
            [loss, Y_init],
            feed_dict=test_feed_dictionary
        )
        save_Y_init[steps] = observed_Y_init
        save_loss[steps] = observed_loss
        if steps % options['logging_frequency'] == 0:
            logging.info(
                'Step: %7u   loss: %12f   Y_0: %12f   Time passed: %12f'
                % (steps,
                   observed_loss,
                   observed_Y_init,
                   np.round((time.time() - start_time), 2)
            ))
        # sampling
        dW_training, X_training = currentProblem.sample(
            options['number_of_training_samples'],
            options['number_of_time_intervals']
        )
        # training
        test_feed_dictionary = {
            dW: dW_training,
            X: X_training,
            is_getting_trained: True
        }
        # more logging
        if options['produce_summary']:
            summary, _ = session.run(
                [merged, training_operations],
                feed_dict=test_feed_dictionary
            )
            train_writer.add_summary(summary, steps)
        else:
            session.run(
                training_operations,
                feed_dict=test_feed_dictionary
            )
    return save_Y_init, save_loss

if __name__ == '__main__':
    options = default_options.copy()
    print('Choose Problem to solve: '
          '\n- Input 0 for: Classical Black Scholes Model'
          '\n- Input 1 for: 10 Dim Basket Option'
          '\n- Input 2 for: 10 Dim Rainbow Option'
          '\n- Input 3 for: 100 Dim Basket Option'
          '\n- Input 4 for: 100 Dim Rainbow Option'
          '\n- Input 5 for: 30 dim Dax Stock Basket '
          '\n- Input 6 for: Counterparty Credit Risk with 1 dimension'
          '\n- Input 7 for: Counterparty Credit Risk with 100 dimension'
          '\n- Input 8 for: Hamiliton Jacobi Bellman Equation'
          )
    input_problem = input("Input number of problem and hit enter:")
    if(input_problem == "0"):
        problem_to_solve = 'ClassicalBS'
        problem_name = 'Classical Black Scholes Model'
        estimated_solution = 1.9566
    elif (input_problem == "1"):
        problem_to_solve = 'BasketOption10'
        problem_name = 'Basket option with 10 underlyings'
        estimated_solution = 3.45763
    elif (input_problem == "2"):
        problem_to_solve = 'RainbowOption10'
        problem_name = 'Rainbow option with 10 underlyings'
        estimated_solution = 16.410989
    elif (input_problem == "3"):
        problem_to_solve = 'BasketOption100'
        problem_name = 'Basket option with 100 underlyings'
        estimated_solution = None
    elif (input_problem == "4"):
        problem_to_solve = 'RainbowOption100'
        problem_name = 'Rainbow option with 100 underlyings'
        estimated_solution = None
    elif (input_problem == "5"):
        problem_to_solve = 'Dax_Basket'
        problem_name = 'Basket option on the Dax'
        estimated_solution = None
    elif (input_problem == "6"):
        problem_to_solve = 'Counterparty_CreditRisk1'
        problem_name = 'Black Scholes with counterparty credit risk on 1 dimension'
        estimated_solution = -0.884
        # estimated_solution = -1.013
    elif (input_problem == "7"):
        problem_to_solve = 'Counterparty_CreditRisk100'
        problem_name = 'Black Scholes with counterparty credit risk on 100 dimension'
        estimated_solution = 2.626
    elif (input_problem == "8"):
        problem_to_solve = 'HJB'
        problem_name = 'Hamilton Jacobi Bellman equation'
        estimated_solution = 4.5901
    else:
        sys.exit("This is not valid input")

    print('\n\n----- Start solving problem %s -----\n\n' % problem_name)

    ### logging intro
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(levelname)-6s %(message)s'
    )

    # Command line flags
    FLAGS = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_string(
        'problem_name',
        '',
        'Name of problem to solve'
    )
    tf.app.flags.DEFINE_string(
        'summaries_directory',
        'Users/chris/PycharmProjects/DeepNN-Solver/Summaries',
        'Where to store summaries'
    )
    predicate = lambda member: inspect.isclass(member)                 \
                               and issubclass(member, myproblems.myProblems) \
                               and member != myproblems.myProblems
    try:
        # Select problem
        problem_class = getattr(sys.modules['myproblems'], problem_to_solve)
        if not predicate(problem_class): raise AttributeError
    except AttributeError:
        # Print usage
        print("Did you misstype? This is not a problem to solve..")
    else:
        start_time = time.time()
        # Create setup for Backtesting
        currentRun = 1
        plotsave_Y_init = np.empty([options['testing_samplerange'] + 1, options['number_of_iterations'] + 1])
        plotsave_loss = np.empty([options['testing_samplerange'] + 1, options['number_of_iterations'] + 1])
        plotsave_Y_init[0, :] = list(range(0, options['number_of_iterations'] + 1))
        plotsave_loss[0, :] = list(range(0, options['number_of_iterations'] + 1))
        while currentRun <= options['testing_samplerange']:
            # Solve problem
            logging.info(
                'Run: %u of %u'
                % (currentRun, options['testing_samplerange'])
            )
            session = tf.Session()
            plotsave_Y_init[currentRun, :], plotsave_loss[currentRun, :] = solve(problem_class(), session=session)
            session.close()
            tf.reset_default_graph()
            logging.info(
                'Session: %u is closed. Resulting Approximation is %8f'
                % (currentRun, plotsave_Y_init[currentRun, -1])
            )
            currentRun += 1
        logging.disable(sys.maxsize)
        print("\n\n----- Problem has been solved in %5f secounds -----" % (np.round((time.time() - start_time), 2)))


        ###
        ### Creating Plots
        ###
        ################ Plot 1 loss with logy #################

        plotsave_loss_mean = np.mean(plotsave_loss[1:,:], axis=0)
        plotsave_loss_mean = gaussian_filter1d(plotsave_loss_mean, sigma=1)

        plt.plot(plotsave_loss[0,:], plotsave_loss_mean, color="#073642")
        plt.fill_between(
            plotsave_loss[0,:],
            gaussian_filter1d(np.max(plotsave_loss[1:,:], axis=0), sigma=1),
            gaussian_filter1d(np.min(plotsave_loss[1:,:], axis= 0), sigma=1),
            color='#4472c4',
            alpha=.8
        )
        ax = plt.subplot(111)
        # Remove the plot frame lines. They are unnecessary chartjunk.
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # Ensure that the axis ticks only show up on the bottom and left of the plot.
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.set_yscale("log")
        plt.xlim(1, options['number_of_iterations'])
        plt.title("Decrease of loss function over number of iterations", pad = 9, fontsize=16)
        ax.set_xlabel("Number of iterations", labelpad=4, fontsize=14, color="#333533");
        ax.set_ylabel("Loss function", labelpad=4, fontsize=14, color="#333533");
        ax.set_facecolor("#f2f2f2")
        ax.set_axisbelow(True)
        plt.grid(True, color="#93a1a1", alpha=0.2)

        ################ Plot 2 loss with log log #################

        plt.figure()
        ax = plt.subplot(111)
        ax.plot(plotsave_loss[0,:], gaussian_filter1d(plotsave_loss_mean, sigma=1) , color="#073642")
        ax.fill_between(
            plotsave_loss[0,:],
            gaussian_filter1d(np.max(plotsave_loss[1:,:], axis=0), sigma=1),
            gaussian_filter1d(np.min(plotsave_loss[1:,:], axis= 0), sigma=1),
            color='#4472c4',
            alpha=.8)
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        plt.xlim(1, options['number_of_iterations'])
        plt.title("Decrease of loss function over number of iterations", pad=7, fontsize=16)
        ax.set_xlabel("Number of iterations", labelpad=4, fontsize=14, color="#333533");
        ax.set_ylabel("Loss function", labelpad=4, fontsize=14, color="#333533");
        ax.set_facecolor("#f2f2f2")
        ax.set_axisbelow(True)
        plt.grid(True, color="#93a1a1", alpha=0.2)

        if(estimated_solution != None):
            ### Plot 3 loss with ylog
            plt.figure()
            ax = plt.subplot(111)
            plotsave_error = np.abs(plotsave_Y_init[1:, :] - estimated_solution)
            plotsave_error_mean = np.mean(plotsave_error, axis=0)
            plotsave_error_mean = gaussian_filter1d(plotsave_error_mean, sigma=1)
            ax.plot(plotsave_loss[0, :], plotsave_error_mean, color="#073642", label="plot1")
            ax.fill_between(
                plotsave_loss[0, :],
                gaussian_filter1d(np.max(plotsave_error, axis=0), sigma=1),
                gaussian_filter1d(np.min(plotsave_error, axis=0), sigma=1),
                color='#4472c4',
                alpha=.8)
            ax.set_yscale("log")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()
            plt.xlim(1, options['number_of_iterations'])
            plt.title("$L^1$ Error over number of iterations", pad=9, fontsize=16)
            ax.set_xlabel("Number of iterations", labelpad=4, fontsize=14, color="#333533");
            ax.set_ylabel("$L^1$ Error", labelpad=4, fontsize=14, color="#333533");
            ax.set_facecolor("#f2f2f2")
            ax.set_axisbelow(True)
            plt.grid(True, color="#93a1a1", alpha=0.2)

            ### Plot 4 loss with log log
            plt.figure()
            ax = plt.subplot(111)
            ax.plot(plotsave_loss[0, :], plotsave_error_mean, color="#073642", label="plot1")
            ax.fill_between(
                plotsave_loss[0, :],
                gaussian_filter1d(np.max(plotsave_error, axis=0), sigma=1),
                gaussian_filter1d(np.min(plotsave_error, axis=0), sigma=1),
                color='#4472c4',
                alpha=.8)
            ax.set_yscale("log")
            ax.set_xscale("log")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()
            plt.xlim(1, options['number_of_iterations'])
            plt.title("$L^1$ Error over number of iterations", pad=9, fontsize=16)
            ax.set_xlabel("Number of iterations", labelpad=4, fontsize=14, color="#333533");
            ax.set_ylabel("$L^1$ Error", labelpad=4, fontsize=14, color="#333533");
            ax.set_facecolor("#f2f2f2")
            ax.set_axisbelow(True)
            plt.grid(True, color="#93a1a1", alpha=0.2)

        else:
            ### Plot 5 for Y0 for no reference value with ylog
            plt.figure()
            plotsave_Y_init_mean = np.mean(plotsave_Y_init[1:, :], axis=0)
            plotsave_Y_init_mean = gaussian_filter1d(plotsave_Y_init_mean, sigma=1)
            plt.plot(plotsave_Y_init[0, :], plotsave_Y_init_mean, color="#073642")
            plt.fill_between(
                plotsave_Y_init[0, :],
                gaussian_filter1d(np.max(plotsave_Y_init[1:, :], axis=0), sigma=1),
                gaussian_filter1d(np.min(plotsave_Y_init[1:, :], axis=0), sigma=1),
                color='#4472c4',
                alpha=.8
            )
            ax = plt.subplot(111)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()
            # ax.set_yscale("log")
            plt.xlim(1, options['number_of_iterations'])
            plt.title("Movement of approximation over iterations", pad=9, fontsize=16)
            ax.set_xlabel("Number of iterations", labelpad=4, fontsize=14, color="#333533");
            ax.set_ylabel("Approximation Y0", labelpad=4, fontsize=14, color="#333533");
            ax.set_facecolor("#f2f2f2")
            ax.set_axisbelow(True)
            plt.grid(True, color="#93a1a1", alpha=0.2)

            ################ Plot 6 Y0 with log log #################
            plt.figure()
            ax = plt.subplot(111)
            ax.plot(plotsave_Y_init[0, :], gaussian_filter1d(plotsave_Y_init_mean, sigma=1), color="#073642")
            ax.fill_between(
                plotsave_Y_init[0, :],
                gaussian_filter1d(np.max(plotsave_Y_init[1:, :], axis=0), sigma=1),
                gaussian_filter1d(np.min(plotsave_Y_init[1:, :], axis=0), sigma=1),
                color='#4472c4',
                alpha=.8)
            ax.set_xscale("log")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()
            plt.xlim(1, options['number_of_iterations'])
            plt.title("Movement of approximation over iterations", pad=7, fontsize=16)
            ax.set_xlabel("Number of iterations", labelpad=4, fontsize=14, color="#333533");
            ax.set_ylabel("Approximation Y0", labelpad=4, fontsize=14, color="#333533");
            ax.set_facecolor("#f2f2f2")
            ax.set_axisbelow(True)
            plt.grid(True, color="#93a1a1", alpha=0.2)
        plt.show()