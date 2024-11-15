# feature number
in_dim = 15

num_hidden = 2
num_neuron = 50

lr = 0.02
epoch = 2000
layers_linear = [in_dim+1, 1]
layers_low = [in_dim] + num_hidden*[num_neuron] + [1]
layers_high = [in_dim+num_neuron] + num_hidden*[num_neuron] + [1]
