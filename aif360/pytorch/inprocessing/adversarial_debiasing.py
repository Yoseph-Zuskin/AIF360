try:
    import torch
except ImportError as error:
    from logging import warning
    warning("{}: AdversarialDebiasing will be unavailable. To install the PyTorch version, run:\n"
            "pip install 'aif360[PyTorch]' or pip install 'aif360[All]'".format(error))
import torch.nn as nn
import torch.optim as optim
import numpy as np
from re import sub
from math import ceil
from copy import deepcopy
from aif360.algorithms import Transformer
from sklearn.exceptions import NotFittedError

from inspect import getmembers, isfunction
# Define list of valid PyTorch weight initialization functions
VALID_WEIGHT_INIT = [m[0] for m in getmembers(torch.nn.init, isfunction) if '_' == m[0][-1]]

default_classifier_ann = lambda input_size, hidden_units, p: nn.Sequential(
# This is an example of a simple lambda function which can be used as a model
# argument during instantiation. In this default function, the hidden_units are an
# one-item list, but ot can be a longer list for deep artificial neural networks.
# The same applies to the dropout probability (p) list parameter.
    nn.Linear(input_size, hidden_units[0]),
    nn.ReLU(),
    nn.Dropout(p=p[0]),
    # nn.Linear(hidden_units[0], hidden_units[1]),
    # nn.ReLU(),
    # nn.Dropout(p=p[1]),
    nn.Linear(hidden_units[-1], 1)
) # (int, list, list) -> torch.nn.Sequential

class ClassifierModel(nn.Module):
    r"""Dynamic classifier model instantiation class.
    Author: Yoseph Zuskin
    """
    
    def __init__(self, layers, output_activation):
        r"""
        Args:
            layers (torch.nn.Sequential): PyTorch modules sequence which defines the
                composition of the classifier`s artificial neural network architecture.
            output_activation (torch.nn.functional.*): Activation function to be
                applied on output layer to generate predictions.
            
        """
        super(ClassifierModel, self).__init__()
        if type(layers) == nn.Sequential:
            self.ann = layers 
        else:
            raise ValueError("Must enter layers as a valid torch.nn.Sequential object.")
        self.output_activation = output_activation
    
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
    def __init__(self, layers, constant=1.0):
        r"""
        Args:
            layers (torch.nn.Sequential): PyTorch modules sequence which defines the
                composition of the classifier`s artificial neural network architecture.
            constant (int, float, optional): Constant weight variable. Default is 1.0.
        """
        super(AdversaryModel, self).__init__()
        if type(layers) == nn.Sequential:
            self.ann = layers 
        else:
            raise ValueError("Must enter layers as a valid torch.nn.Sequential object.")
        self.c = torch.tensor(constant, requires_grad=True)
        self.s = nn.Sigmoid() # Define the classifier logit decoder layer
        self.encoder = nn.Linear(3, 1) # Define the adversary linear encoder layer
    
    def forward(self, x, y):
        # type: (ClassifierModel, Tensor, Tensor) -> (Tensor, Tensor)
        #print(self.classifier)
        x = self.ann(x)
        #print(x.grad)
        x = self.s(1.0 + torch.abs(self.c) * x)
        #print(x.grad)
        x = self.encoder(torch.cat([x, x * y, x * (1.0 - y)], dim=1))
        #print(x.grad)
        x_last = torch.sigmoid(x)
        return x_last, x

class StaircaseExponentialLR(optim.lr_scheduler._LRScheduler):
    r"""Decays the learning rate using a dampened exponential decay process
    with the option to toggle staircase or smooth decay patterns. Based on
    the exp_lr_scheduler custom function published by Tejas Khot on GitHub.
    Source: https://gist.github.com/tejaskhot/2bbc4f15ba7bde33da9aa3a9dcb5c3e0
    """
    
    def __init__(self, optimizer, global_steps, init_lr, decay_steps, decay_rate,
                 lr_clip=None, staircase=True, verbose=True):
        if not isinstance(optimizer, optim.Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        self.global_steps = global_steps
        self.init_lr = init_lr
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.lr_clip = lr_clip
        self.staircase = staircase
        self.verbose = verbose
    
    def step(self, global_step, model_name):
        if self.staircase:
            lr = self.init_lr * self.decay_rate**(global_step // self.decay_steps)
        else:
            lr = self.init_lr * self.decay_rate**(global_step / self.decay_steps)
        if self.lr_clip is not None and type(self.lr_clip) == float:
            lr = max(lr, self.lr_clip)

        if self.verbose and global_step % self.decay_steps == 0:
            model_name = sub(r"(\w)([A-Z])", r"\1 \2", model_name)
            print(f"Learning rate of the {model_name.lower()} is now set to {lr}")

        for param_group in self.optimizer.param_groups:
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
    Author: Yoseph Zuskin
    """

    def __init__(self, unprivileged_groups, privileged_groups, input_size, debias=True,
                 seed=None, classifier=ClassifierModel, adversary=AdversaryModel,
                 num_epochs=50, batch_size=128, adversary_loss_weight=0.1,
                 classifier_args=[default_classifier_ann, torch.sigmoid],
                 classifier_num_hidden_units=[200], classifier_dropout_ps=[0.5],
                 adversary_args=None, device="check", initializer_fn="xavier_uniform_",
                 classifier_loss_fn=nn.BCELoss(reduction="mean"),
                 adversary_loss_fn=nn.BCELoss(reduction="mean"),
                 classifier_optimizer=optim.Adam, adversary_optimizer=optim.Adam,
                 init_lr=0.001, decay_steps=1000, decay_rate=0.96, lr_clip=None,
                 staircase=True, verbose=True, *args, **kwargs):
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
                initializer_fn (str, torch.nn.init.*_, optional): Chosen weight initialization
                    function. Must be a function with a name ending with ``_``. Default is
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
            
            # Define the unprivileged and privileged groups
            if len(unprivileged_groups) > 1 or len(privileged_groups) > 1:
                raise ValueError("Only one unprivileged_group or privileged_group supported.")
            self.unprivileged_groups = unprivileged_groups
            self.privileged_groups = privileged_groups
            self.protected_attribute_name = list(self.unprivileged_groups[0].keys())[0]
            
            # Define the random seed for replicability of outputs
            self.seed = seed
            if seed is not None and type(seed) == int and seed >=0:
                torch.manual_seed(seed)
            
            # Declare device setting for loading Tensors onto CPU or GPU memory
            if device is None or device.lower() in ["check", "cuda"]:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            elif device.lower() == "cuda:0":
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            elif device.lower() == "cpu":
                device = torch.device("cpu")
            else:
                raise ValueError("Device setting choice is invalid. Must be 'cuda', "+
                                 "'cuda:0', 'cpu', or (the default of) 'check'.")
            self.device = device
            
            # Define and validate the specified weight initialization function
            if callable(initializer_fn) and initializer_fn.__name__ in VALID_WEIGHT_INIT:
                self.initializer = initializer_fn
            elif type(initializer_fn) == str and initializer_fn in VALID_WEIGHT_INIT:
                self.initializer = getattr(torch.nn.init, initializer_fn)
            else:
                raise ValueError(f"The chosen {initializer_fn} weights initialization function is invalid.")
            
            # Validate the model(s) instantiation parameters
            if type(input_size) != int or input_size <= 0:
                raise ValueError(f"Input size of {input_size} is invalid. It must be passed as a positive ingeter value.")
            if type(classifier_num_hidden_units) != list or all(x <= 0 and type(x) == int for x in classifier_num_hidden_units):
                raise ValueError(f"Number(s) of hidden units list of {classifier_num_hidden_units} " + \
                                 "is invalid. All values within the list must be positive integers.")
            if type(classifier_dropout_ps) != list or not all(0 < x < 1 and type(x) == float for x in classifier_dropout_ps):
                raise ValueError(f"Dropout value(s) list of {classifier_dropout_ps} is invalid. " + \
                                 "All values within the list must be floating point numbers between 0 and 1.")
            
            # Instanciate the classifier and (if debias is set to True) adversary models
            if classifier_args is not None and type(classifier_args) == list:
                ann = classifier_args[0](input_size, classifier_num_hidden_units, classifier_dropout_ps)
                classifier_args = [ann] + classifier_args[1:]
                classifier = classifier(*classifier_args)
                self.classifier = classifier.to(device)
                if debias:
                    adversary = adversary(layers=deepcopy(ann)).to(device) \
                    if adversary_args is None else adversary(layers=ann, *adversary_args)
                    self.adversary = adversary.to(device)
            else: # This enables users to input a pre-instantiate classifier and adversary models
                self.classifier = classifier.to(self.device)
                if self.debias:
                    self.adversary = adversary.to(self.device)
            
            # Validate optimizer instantiation parameters and instantiate the optimizer(s)
            if type(init_lr) != float or init_lr <= 0 or init_lr >= 1:
                raise ValueError(f"Initial learning rate parameter of {init_lr} is invalid. Must be a floating point number between 0 and 1.")
            else:
                self.init_lr = init_lr
            if type(decay_steps) != int or decay_steps <= 0:
                raise ValueError(f"Learning rate decay steps parameter of {decay_steps} is invalid. Must be a positive integer.")
            else:
                self.decay_steps = decay_steps
            if type(decay_rate) != float or decay_rate <= 0 or decay_rate >= 1:
                raise ValueError(f"Learning rate decay parameter of {decay_rate} is invalid. Must be a floating point number between 0 and 1.")
            else:
                self.decay_rate = decay_rate
            if lr_clip is not None or type(lr_clip) == float and (lr_clip <= 0 or lr_clip >= 1):
                raise ValueError(f"Learning rate decay clip parameter of {lr_clip} is invalid. Must be a floating point number between 0 and 1, or None.")
            else:
                self.lr_clip = lr_clip
            if type(staircase) != bool or type(staircase) == int and (staircase != 0 or staircase != 1):
                raise ValueError(f"Staircase decay option parameter of {staircase} is invalid. Must be a boolean, 0, or 1.")
            else: 
                self.staircase = staircase
            
            # Define the loss function criteria & tracking list for model(s)
            self.classifier_criterion = classifier_loss_fn
            self.adversary_criterion = adversary_loss_fn
            
            # Define parameters related to the model(s) fitting process
            self.debias = debias
            self.adversary_loss_weight = adversary_loss_weight ### The paper implements a different approach (np.sqrt(1/global_step)), see page 6 ###
            self.num_epochs = num_epochs
            self.batch_size = batch_size  
            
            # Define lists to keep track of the fitting progress
            self.classifier_losses = []
            self.adversary_losses = []
            
            # Parameters related to optimization and learning rate decay
            self.classifier_optim = classifier_optimizer([p for p in classifier.parameters() if p.requires_grad], lr=init_lr)
            self.adversary_optim = adversary_optimizer([p for p in classifier.parameters() if p.requires_grad], lr=init_lr)
            
            # Toggle printing of interim results
            self.verbose = verbose

    def init_weights(self, layer):
        r"""Initialize layer weights and biases if it has any and the chosen initializer
        is valid. Can be applied on any layer and will only initialize parametric layers.
        """
        
        try:
            layer.__getattr__('weight')
            _has_weight = True
        except:
            _has_weight = False
            
        try:
            layer.__getattr__('bias')
            _has_bias = True
        except:
            _has_bias = False
            
        if _has_weight:
            self.initializer(layer.weight.data)
        if _has_bias:
            try:
                self.initializer(layer.bias.data)
            except:
                layer.bias.data.fill_(0.01)
        else:
            pass

    def fit(self, dataset):
        r"""Compute the model parameters of the fair classifier using gradient
        descent.
        Args:
            dataset (BinaryLabelDataset): Dataset containing true labels.
        Returns:
            AdversarialDebiasing: Returns self.
        """
        
        # Set random seed
        if self.seed is not None:
            torch.manual_seed(self.seed)
        
        # Map the dataset labels to 0 and 1.
        temp_labels = dataset.labels.copy()
        temp_labels[(dataset.labels == dataset.favorable_label).ravel(),0] = 1.0
        temp_labels[(dataset.labels == dataset.unfavorable_label).ravel(),0] = 0.0
        
        dataset.labels = temp_labels.copy()
        
        # Define the number of samples, features, global training steps
        num_train_samples = len(dataset.features)
        global_steps = self.num_epochs * ceil(num_train_samples / self.batch_size)
        
        # Create training dataset and loader objects
        protected_attribute_index = dataset.protected_attribute_names.index(self.protected_attribute_name)
        train_dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(dataset.features).float().to(self.device),
            torch.from_numpy(dataset.labels).float().to(self.device),
            torch.from_numpy(dataset.protected_attributes[:, protected_attribute_index].\
            reshape(dataset.protected_attributes.shape[0], -1)).float().to(self.device)
        )
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
                 
        # Setup staircase-wise exponentially decaying learning rate schedulers
        classifier_lr_scheduler = StaircaseExponentialLR(self.classifier_optim, global_steps, self.init_lr, self.decay_steps,
                                                         self.decay_rate, self.lr_clip, self.staircase, self.verbose)
        classifier_lr_scheduler.initial_lr = self.init_lr
        if self.debias:
            adversary_lr_scheduler = StaircaseExponentialLR(self.adversary_optim, global_steps, self.init_lr, self.decay_steps,
                                                            self.decay_rate, self.lr_clip, self.staircase, self.verbose)
            adversary_lr_scheduler.initial_lr = self.init_lr
                
        # Define unit vector normalization function for adversary-to-classifier gradient projection
        normalize = lambda x: x / (torch.norm(x) + np.finfo(np.float32).tiny)
        
        # Begin to train both models over each epoch
        global_step = 0
        if self.verbose:
            print(f'Starting to train model(s) on {self.device.type}:')
        #classifier.train()
        #if self.debias:
        #    adversary.train()
        for epoch in range(self.num_epochs):
            if self.debias:
                for i, data in enumerate(train_loader, 0):
                    # Update learning rates
                    classifier_lr_scheduler.step(global_step, self.classifier.__class__.__name__)
                    adversary_lr_scheduler.step(global_step, self.adversary.__class__.__name__)
                    # Train the classifier model
                    self.classifier.zero_grad()
                    batch_features = data[:][0].to(self.device)
                    batch_labels = data[:][1].to(self.device)
                    pred_labels, pred_logits = self.classifier(batch_features)
                    classifier_error = self.classifier_criterion(pred_labels, batch_labels)
                    self.classifier_losses.append(classifier_error.item())
                    classifier_mean_error = np.mean(self.classifier_losses)
                    classifier_error.backward()
                    # Update the parameters for the classifier layers within the adversary model
                    c_params, a_params = dict(self.classifier.named_parameters()), dict(self.adversary.named_parameters())
                    for (c_p, a_p) in zip(c_params.values(), a_params.values()): # This zip acts like an inner join for parameters
                        a_p.data = deepcopy(c_p.data)
                    # Train the adversary
                    self.adversary.zero_grad()
                    batch_protected_attributes_labels = data[:][2].to(self.device)
                    pred_protected_attributes_labels, pred_protected_attributes_logits = self.adversary(
                    batch_features, batch_labels)
                    adversary_error = self.adversary_criterion(pred_protected_attributes_labels, batch_protected_attributes_labels)
                    adversary_error.backward()
                    self.adversary_losses.append(adversary_error.item())
                    adversary_mean_error = np.mean(self.adversary_losses)
                    # Adjust the classifier's gradients according to the normnalized adversary gradients
                    c_params, a_params = dict(self.classifier.named_parameters()), dict(self.adversary.named_parameters())
                    for p in c_params:
                        unit_adversary_grad = normalize(a_params[p].grad)
                        c_params[p].grad -= torch.sum((c_params[p].grad * unit_adversary_grad))
                        c_params[p].grad -= self.adversary_loss_weight * a_params[p].grad
                    self.adversary_optim.step() # Update adversary model parameters
                    self.classifier_optim.step() # Update classifier model parameters
                    if i % 200 == 0:
                        print("Epoch: [%d/%d] Batch: [%d/%d]\tClassifier_Loss: %.4f\tAdversary Loss: %.4f\tC(x): %.4f\tA(x, y): %.4f" % \
                        (epoch + 1, self.num_epochs, i + 1, len(train_loader), self.classifier_losses[-1],
                    adversary_error.item(), classifier_mean_error, adversary_mean_error))
                    global_step += 1
            else:
                for i, data in enumerate(train_loader, 0):
                    # Update learning rates
                    classifier_lr_scheduler.step(global_step, self.classifier.__class__.__name__)
                    # Train the classifier model
                    self.classifier.zero_grad()
                    batch_features = data[:][0].to(self.device)
                    batch_labels = data[:][1].to(self.device)
                    pred_labels, pred_logits = self.classifier(batch_features)
                    classifier_error = self.classifier_criterion(pred_labels, batch_labels)
                    classifier_error.backward()
                    self.classifier_losses.append(classifier_error.item())
                    classifier_mean_error = np.mean(self.classifier_losses)
                    self.classifier_optim.step() # Update classifier model parameters
                    # Print training statistics if verbose option is set to True
                    if self.verbose and i % 200 == 0:
                        print("Epoch: [%d/%d] Batch: [%d/%d]\tClassifier Loss: %.4f\tC(x): %.4f" % \
                        (epoch + 1, self.num_epochs, i + 1, len(train_loader), self.classifier_losses[-1], classifier_mean_error))
                    global_step += 1
        self.classifier.training = False
        if self.debias:
            self.adversary.training = False
        return self

    def predict(self, dataset):
        r"""Obtain the predictions for the provided dataset using the fair
        classifier learned.
        Args:
            dataset (BinaryLabelDataset): Dataset containing labels that needs
                to be transformed.
        Returns:
            dataset (BinaryLabelDataset): Transformed dataset.
        """
        
        if self.classifier.training:
            raise NotFittedError("The AdversarialDebiasing fit method must first be executed before using the predict method.")
        
        if self.seed is not None:
            torch.manual_seed(self.seed)

        features = torch.from_numpy(dataset.features).float().to(self.device)
        
        pred_labels = self.classifier.forward(features)[0]
        pred_labels = pred_labels.cpu().detach().numpy().tolist()
        
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