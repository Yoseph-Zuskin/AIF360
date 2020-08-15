import numpy as np
from aif360.algorithms import Transformer
try:
    import torch
except ImportError as error:
    from logging import warning
    warning("{}: AdversarialDebiasing will be unavailable. To install the PyTorch version, run:\n"
            "pip install 'aif360[PyTorchAdversarialDebiasing]'".format(error))
import torch.nn as nn
import torch.optim as optim

default_classifier_ann = lambda input_size, hidden_units, p: nn.Sequential(
# This is an example of a simple lambda function which can be used as a model
# argument during instantiation. In this default function, the hidden_units are
# a one-item list but can be a longer list for deep artificial neural networks.
# The same applies to the dropout probability parameter.
    nn.Linear(input_size, hidden_units[0]),
    nn.ReLU(),
    nn.Dropout(p=p[0]),
    # nn.Linear(hidden_units[0], hidden_units[1]),
    # nn.ReLU(),
    # nn.Dropout(p=p[1]),
    nn.Linear(hidden_units[-1], 1)
)

class ClassifierModel(nn.Module):
    r"""Dynamic classifier model instantiation class.
    
    Author: Yoseph Zuskin
    """
    
    def __init__(self, layers, output_activation):
        r"""
        Args:
            layers (torch.nn.Sequential): Ordered modules dictionary defining the
                sequence of the classifier"s artificial neural network architecture.
            output_activation (torch.nn.functional.*): Activation function to be
                applied on output layer to generate predictions.
            
        """
        super(ClassifierModel, self).__init__()
        
        if type(layers) == nn.Sequential:
            self.ann = layers 
        else:
            raise ValueError("Must enter layers as a valid torch.nn.Sequential object.")
        self.output_activation = output_activation ### MUST IMPLEMENT ERROR CHECK HERE!
    
    def forward(self, x):
        # type: (Tensor) -> (Tensor, Tensor)
        x = self.ann(x)
        x_last = self.output_activation(x)
        return x_last, x

class AdversaryModel(nn.Module):
    r"""Default adversary model instantiation class. Based on the TensorFlow
    implementation of this library's original AdversarialDebiasing class.
    
    Author: Yoseph Zuskin
    """
    # FUTURE WORK: Implement process for more than 1 protected attributed
    def __init__(self, y):
        super(AdversaryModel, self).__init__()
        # Transform target labels into tensor object
        self.y = torch.from_numpy(y).float()
        # Instantiate constant weight variable
        self.c = torch.tensor(1.0, requires_grad=True)
        # Define the classifier logit decoder layer
        self.s = nn.Sigmoid()
        # Define the adversary linear encoder layer
        self.encoder = nn.Linear(3, 1)
    
    def forward(self, x):
        # type: (Tensor, Tensor) -> (Tensor, Tensor)
        x = self.s(1.0 + torch.abs(self.c) * x)
        x = self.encoder(torch.cat([x, x * self.y, x * (1.0 - self.y)], dim=1))
        x_last = torch.sigmoid(x)
        return x_last, x
        
def exp_lr_scheduler(optimizer, global_step, init_lr, decay_steps, decay_rate,
                    lr_clip=None, staircase=True, verbose=False):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs.
    
    Author: Tejas Khot
    Source: https://gist.github.com/tejaskhot/2bbc4f15ba7bde33da9aa3a9dcb5c3e0
    """
    if staircase:
        lr = init_lr * decay_rate**(global_step // decay_steps)
    else:
        lr = init_lr * decay_rate**(global_step / decay_steps)
    if type(lr_clip) is not None and type(lr_clip) == float:
        lr = max(lr, lr_clip)

    if verbose and global_step % decay_steps == 0:
        print("LR is set to {}".format(lr))

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

class AdversarialDebiasing(Transformer):
    r"""Adversarial debiasing is an in-processing technique that learns a
    classifier to maximize prediction accuracy and simultaneously reduce an
    adversary"s ability to determine the protected attribute from the
    predictions [5]_. This approach leads to a fair classifier as the
    predictions cannot carry any group discrimination information that the
    adversary can exploit.
    References:
        .. [5] B. H. Zhang, B. Lemoine, and M. Mitchell, "Mitigating Unwanted
           Biases with Adversarial Learning," AAAI/ACM Conference on Artificial
           Intelligence, Ethics, and Society, 2018.
    """

    def __init__(self, unprivileged_groups, privileged_groups, debias=True,
                 seed=None, adversary_loss_weight=0.1, num_epochs=50, batch_size=128,
                 classifier=ClassifierModel, adversary=AdversaryModel,
                 classifier_args=[default_classifier_ann, torch.sigmoid],
                 classifier_num_hidden_units=[200], classifier_dropout_ps=[0.5],
                 adversary_args=None, device="check", weights_method="xavier_uniform_",
                 optimizer=optim.Adam, init_lr=0.001, decay_steps=1000, decay_rate=0.96,
                 lr_clip=None, staircase=True, verbose=False, *args, **kwargs):
            r"""
            Args:
                unprivileged_groups (tuple): Representation for unprivileged groups.
                privileged_groups (tuple): Representation for privileged groups.
                debias (bool, optional): Learn a classifier with or without debiasing.
                classifier (nn.Module, optional): Custom classifier model. Will use
                    default artificial neural network if no alternative is specified.
                adversary (nn.Module, optional): Custom adversary model. Will use
                    default artificial neural network if no alternative is specified.
                seed (int, optional): Seed to make `predict` repeatable.
                adversary_loss_weight (float, optional): Hyperparameter that chooses
                    the strength of the adversarial loss.
                num_epochs (int, optional): Number of training epochs.
                batch_size (int, optional): Batch size.
                classifier_args (list, optional): Additional arguments to be used
                    during classifier model instantiation after the input size.
                    First argument must be a function that accepts at least one
                    parameter for the network's input size based on the dataset
                    feature's dimensions. Other arguments could include number of
                    hidden neurons and dropout prabilities for each layer.
                classifier_num_hidden_units (list, optional): List defining the number
                    of hidden perceptron units should exist within the classifier"s
                    hidden layers. Default is [200].
                classifier_dropout_ps (list, optional). List defining the probabilities
                    to be used in each dropout instance in the classifier"s architecture.
                    Default is [0.5].
                adversary_args (list, optional): Additional arguments to be used
                    during classifier model instantiation after the pred_label and
                    pred_logit parameters are passed from classifier model.
                device (str, optional): Device setting, which can be ``cuda`` or
                    ``cpu``. Will ``check`` if cuda is available by default.
                weights_method (str, optional): Weights initialization method choice
                    for all layers in both classifier and adversary models. Default is
                    ``xavier_uniform_``.
                optimizer (torch.optim.*): Optimization method to use for model fitting
                    and exponential decay of the learning rate. Default is optim.Adam.
                init_lr (float, optional): Initial learning rate. Default is 0.001.
                decay_steps (int, optional): Number of decay steps. Default is 1000.
                decay_rate (float, optional): Rate of exponential decay for the learning
                    rate during training. Must be more than 0 and les than 1. Default is
                    0.96.
                lr_clip (float, optional): Minimum limit for the learning rate's exponential
                    decay during training. Default is None.
                staircase (bool, optional): Option to toggle staircase-wise exponential
                    decay of learning rate during training. Default is True.
                verbose (bool, optional): Option to toggle printing of interim progress.
                    Default is True.
            """
            
            super(AdversarialDebiasing, self).__init__(
                unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups)
            # Define the random seed for replicability of outputs
            self.seed = seed
            # Define the unprivileged and privileged groups
            self.unprivileged_groups = unprivileged_groups
            self.privileged_groups = privileged_groups
            if len(self.unprivileged_groups) > 1 or len(self.privileged_groups) > 1:
                raise ValueError("Only one unprivileged_group or privileged_group supported.")
            self.protected_attribute_name = list(self.unprivileged_groups[0].keys())[0]
            # Define the models for this instance
            self._classifier_model = classifier
            self._adversary_model = adversary
            # Parameters related to the models
            self.adversary_loss_weight = adversary_loss_weight
            self.num_epochs = num_epochs
            self.batch_size = batch_size
            self.debias = debias
            self.classifier_args = classifier_args
            self.adversary_args = adversary_args
            self.classifier_num_hidden_units = classifier_num_hidden_units
            self.classifier_dropout_ps = classifier_dropout_ps
            if type(weights_method) == str:
                self.weights_method = weights_method.lower().replace("_","").replace(" ","").replace("-","")
            else:
                raise ValueError("The specified method must be entered as a string.")
            if self.weights_method.lower() not in ["uniform", "normal", "xavieruniform", "xaviernormal",
                                                  "kaiminguniform", "kaimingnormal"]:
                raise ValueError("Invalid weights initializer choice.")
            # Parameters related to optimization and learning rate decay
            self.optimizer = optimizer
            self.init_lr = init_lr
            self.decay_steps = decay_steps
            self.decay_rate = decay_rate
            self.lr_clip = lr_clip
            self.staircase = staircase
            # Toggle printing of interim results
            self.verbose = verbose
            
            # Declare device setting for loading Tensors onto CPU or GPU memory
            if device is None or device.lower() in ["check", "cuda"]:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            elif device.lower() == "cuda:0":
                elf.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            elif device.lower() == "cpu":
                self.device = torch.device("cpu")
            else:
                raise ValueError("Device setting choice is invalid. Must be 'cuda', "+
                                 "'cuda:0', 'cpu', or (the default of) 'check'.")

    def weights_init(self, m):
        r"""Initialize model wieghts with defined method (Uniform, Xavier, or Kaiming)
        """
        if self.weights_method.lower() == "uniform":
            nn.init.uniform_(m.weight.data)
            if m.bias is not None:
                nn.init.uniform_(m.bias.data)
        elif self.weights_method.lower() == "normal":
            nn.init.normal_(m.weight.data)
            if m.bias is not None:
                nn.init.normal_(m.bias.data)
        elif self.weights_method.lower() == "xavieruniform":
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                nn.init.xavier_uniform_(m.bias.data)
        elif self.weights_method.lower() == "xaviernormal":
            nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.xavier_normal_(m.bias.data)
        elif self.weights_method.lower() == "kaiminguniform":
            nn.init.kaiming_uniform_(m.weight.data)
            if m.bias is not None:
                nn.init.kaiming_uniform_(m.bias.data)
        elif self.weights_method.lower() == "kaimingnormal":
            nn.init.kaiming_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.kaiming_normal_(m.bias.data)

    def fit(self, dataset):
        r"""Compute the model parameters of the fair classifier using gradient
        descent.
        Args:
            dataset (BinaryLabelDataset): Dataset containing true labels.
        Returns:
            AdversarialDebiasing: Returns self.
        """
        
        # Map the dataset labels to 0 and 1.
        temp_labels = dataset.labels.copy()

        temp_labels[(dataset.labels == dataset.favorable_label).ravel(),0] = 1.0
        temp_labels[(dataset.labels == dataset.unfavorable_label).ravel(),0] = 0.0

        # Define the number of samples and features
        num_train_samples, features_dim = np.shape(dataset.features)
        
        # Set random seed
        if type(self.seed) is not None:
            torch.manual_seed(self.seed)
        
        # Instanciate the classifier and (if debias is set to True) adversary models
        if type(self.classifier_args) is not None:
            self.classifier_args = [classifier_args[0](features_dim, self.classifier_num_hidden_units,
                                                       self.classifier_dropout_ps) ] + self.classifier_args[1:]
        classifier = self._classifier_model(*classifier_args).to(device)
        classifier = classifier.apply(self.weights_init)
        if self.debias:
            adversary = self._adversary_model() if type(self.adversary_args
            ) is None else self._adversary_model(*self.adversary_args)
            adversary = adversary.apply(self.weights_init)
        
        # Convert features into Tensor object
        features = torch.from_numpy(dataset.features.float(), requires_grad=False)
        
        # Setup optimizers with exponentially decaying learning rate schedulers
        classifier_optim = self.optimizer(classifier.parameters(), lr=self.init_lr)
        adversary_optim = self.optimizer(adversary.parameters(), lr=self.init_lr)
        classifier_lr_scheduler = exp_lr_scheduler(classifier_optim, epoch, self.init_lr, self.decay_steps,
                                                   self.decay_rate, self.lr_clip, self.staircase, self.verbose)
        adversary_lr_scheduler = exp_lr_scheduler(adversary_optim, epoch, self.init_lr, self.decay_steps,
                                                   self.decay_rate, self.lr_clip, self.staircase, self.verbose)
        
        # Define the loss function criteria & tracking list for model(s)
        criterion = nn.BCELoss(reduction="mean")
        
        # Define lists to keep track of the fitting progress
        classifier_losses, adversary_losses, iters = [], [], 0
        
        # Begin to train both models over each epoch
        if self.verbose:
            print(f"Starting to train model(s) on {self.device}:")
        #for epoch in range(self.num_epochs):
            
    
    def old_fit(self, dataset):
        """Compute the model parameters of the fair classifier using gradient
        descent.
        Args:
            dataset (BinaryLabelDataset): Dataset containing true labels.
        Returns:
            AdversarialDebiasing: Returns self.
        """
        if self.seed is not None:
            np.random.seed(self.seed)
        ii32 = np.iinfo(np.int32)
        self.seed1, self.seed2, self.seed3, self.seed4 = np.random.randint(ii32.min, ii32.max, size=4)

        # Map the dataset labels to 0 and 1.
        temp_labels = dataset.labels.copy()

        temp_labels[(dataset.labels == dataset.favorable_label).ravel(),0] = 1.0
        temp_labels[(dataset.labels == dataset.unfavorable_label).ravel(),0] = 0.0

        with tf.variable_scope(self.scope_name):
            num_train_samples, self.features_dim = np.shape(dataset.features)

            # Setup placeholders
            self.features_ph = tf.placeholder(tf.float32, shape=[None, self.features_dim])
            self.protected_attributes_ph = tf.placeholder(tf.float32, shape=[None,1])
            self.true_labels_ph = tf.placeholder(tf.float32, shape=[None,1])
            self.keep_prob = tf.placeholder(tf.float32)

            # Obtain classifier predictions and classifier loss
            self.pred_labels, pred_logits = self._classifier_model(self.features_ph, self.features_dim, self.keep_prob)
            pred_labels_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.true_labels_ph, logits=pred_logits))
            # torch.mean is the equivalent function

            if self.debias:
                # Obtain adversary predictions and adversary loss
                pred_protected_attributes_labels, pred_protected_attributes_logits = self._adversary_model(pred_logits, self.true_labels_ph)
                pred_protected_attributes_loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(labels=self.protected_attributes_ph, logits=pred_protected_attributes_logits))

            # Setup optimizers with learning rates
            global_step = tf.Variable(0, trainable=False)
            starter_learning_rate = 0.001
            learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                       1000, 0.96, staircase=True)
            classifier_opt = tf.train.AdamOptimizer(learning_rate)
            if self.debias:
                adversary_opt = tf.train.AdamOptimizer(learning_rate)

            classifier_vars = [var for var in tf.trainable_variables() if "classifier_model" in var.name]
            if self.debias:
                adversary_vars = [var for var in tf.trainable_variables() if "adversary_model" in var.name]
                # Update classifier parameters
                adversary_grads = {var: grad for (grad, var) in adversary_opt.compute_gradients(pred_protected_attributes_loss,
                                                                                      var_list=classifier_vars)}
            normalize = lambda x: x / (tf.norm(x) + np.finfo(np.float32).tiny)

            classifier_grads = []
            for (grad,var) in classifier_opt.compute_gradients(pred_labels_loss, var_list=classifier_vars):
                if self.debias:
                    unit_adversary_grad = normalize(adversary_grads[var])
                    grad -= tf.reduce_sum(grad * unit_adversary_grad) * unit_adversary_grad
                    grad -= self.adversary_loss_weight * adversary_grads[var]
                classifier_grads.append((grad, var))
            classifier_minimizer = classifier_opt.apply_gradients(classifier_grads, global_step=global_step)

            if self.debias:
                # Update adversary parameters
                with tf.control_dependencies([classifier_minimizer]):
                    adversary_minimizer = adversary_opt.minimize(pred_protected_attributes_loss, var_list=adversary_vars)#, global_step=global_step)

            self.sess.run(tf.global_variables_initializer())
            self.sess.run(tf.local_variables_initializer())

            # Begin training
            for epoch in range(self.num_epochs):
                shuffled_ids = np.random.choice(num_train_samples, num_train_samples, replace=False)
                for i in range(num_train_samples//self.batch_size):
                    batch_ids = shuffled_ids[self.batch_size*i: self.batch_size*(i+1)]
                    batch_features = dataset.features[batch_ids]
                    batch_labels = np.reshape(temp_labels[batch_ids], [-1,1])
                    batch_protected_attributes = np.reshape(dataset.protected_attributes[batch_ids][:,
                                                 dataset.protected_attribute_names.index(self.protected_attribute_name)], [-1,1])

                    batch_feed_dict = {self.features_ph: batch_features,
                                       self.true_labels_ph: batch_labels,
                                       self.protected_attributes_ph: batch_protected_attributes,
                                       self.keep_prob: 0.8}
                    if self.debias:
                        _, _, pred_labels_loss_value, pred_protected_attributes_loss_vale = self.sess.run([classifier_minimizer,
                                       adversary_minimizer,
                                       pred_labels_loss,
                                       pred_protected_attributes_loss], feed_dict=batch_feed_dict)
                        if i % 200 == 0:
                            print("epoch %d; iter: %d; batch classifier loss: %f; batch adversarial loss: %f" % (epoch, i, pred_labels_loss_value,
                                                                                     pred_protected_attributes_loss_vale))
                    else:
                        _, pred_labels_loss_value = self.sess.run(
                            [classifier_minimizer,
                             pred_labels_loss], feed_dict=batch_feed_dict)
                        if i % 200 == 0:
                            print("epoch %d; iter: %d; batch classifier loss: %f" % (
                            epoch, i, pred_labels_loss_value))
        return self

    def predict(self, dataset):
        """Obtain the predictions for the provided dataset using the fair
        classifier learned.
        Args:
            dataset (BinaryLabelDataset): Dataset containing labels that needs
                to be transformed.
        Returns:
            dataset (BinaryLabelDataset): Transformed dataset.
        """

        if self.seed is not None:
            np.random.seed(self.seed)

        num_test_samples, _ = np.shape(dataset.features)

        samples_covered = 0
        pred_labels = []
        while samples_covered < num_test_samples:
            start = samples_covered
            end = samples_covered + self.batch_size
            if end > num_test_samples:
                end = num_test_samples
            batch_ids = np.arange(start, end)
            batch_features = dataset.features[batch_ids]
            batch_labels = np.reshape(dataset.labels[batch_ids], [-1,1])
            batch_protected_attributes = np.reshape(dataset.protected_attributes[batch_ids][:,
                                         dataset.protected_attribute_names.index(self.protected_attribute_name)], [-1,1])

            batch_feed_dict = {self.features_ph: batch_features,
                               self.true_labels_ph: batch_labels,
                               self.protected_attributes_ph: batch_protected_attributes,
                               self.keep_prob: 1.0}

            pred_labels += self.sess.run(self.pred_labels, feed_dict=batch_feed_dict)[:,0].tolist()
            samples_covered += len(batch_features)

        # Mutated, fairer dataset with new labels
        dataset_new = dataset.copy(deepcopy = True)
        dataset_new.scores = np.array(pred_labels, dtype=np.float64).reshape(-1, 1)
        dataset_new.labels = (np.array(pred_labels)>0.5).astype(np.float64).reshape(-1,1)


        # Map the dataset labels to back to their original values.
        temp_labels = dataset_new.labels.copy()

        temp_labels[(dataset_new.labels == 1.0).ravel(), 0] = dataset.favorable_label
        temp_labels[(dataset_new.labels == 0.0).ravel(), 0] = dataset.unfavorable_label

        dataset_new.labels = temp_labels.copy()

        return dataset_new
