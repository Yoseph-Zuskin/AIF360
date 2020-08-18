import numpy as np
from re import sub
from math import ceil
from aif360.algorithms import Transformer
from sklearn.exceptions import NotFittedError
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
    def __init__(self, c=1.0):
        super(AdversaryModel, self).__init__()
        # Instantiate constant weight variable
        self.c = torch.tensor(c, requires_grad=True)
        # Define the classifier logit decoder layer
        self.s = nn.Sigmoid()
        # Define the adversary linear encoder layer
        self.encoder = nn.Linear(3, 1)
    
    def forward(self, x, y):
        # type: (Tensor, Tensor) -> (Tensor, Tensor)
        x = self.s(1.0 + torch.abs(self.c) * x)
        x = self.encoder(torch.cat([x, x * y, x * (1.0 - y)], dim=1))
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
        model_name = sub(r"(\w)([A-Z])", r"\1 \2", model_name)
        if self.staircase:
            lr = self.init_lr * self.decay_rate**(global_step // self.decay_steps)
        else:
            lr = self.init_lr * self.decay_rate**(global_step / self.decay_steps)
        if self.lr_clip is not None and type(self.lr_clip) == float:
            lr = max(lr, self.lr_clip)

        if self.verbose and global_step % self.decay_steps == 0:
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
    """

    def __init__(self, unprivileged_groups, privileged_groups, debias=True,
                 seed=None, adversary_loss_weight=0.1, num_epochs=50, batch_size=128,
                 classifier=ClassifierModel, adversary=AdversaryModel,
                 classifier_args=[default_classifier_ann, torch.sigmoid],
                 classifier_num_hidden_units=[200], classifier_dropout_ps=[0.5],
                 adversary_args=None, device="check", weights_method="xavier_uniform_",
                 optimizer=optim.Adam, init_lr=0.001, decay_steps=1000, decay_rate=0.96,
                 lr_clip=None, staircase=True, verbose=True, *args, **kwargs):
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
            if len(unprivileged_groups) > 1 or len(privileged_groups) > 1:
                raise ValueError("Only one unprivileged_group or privileged_group supported.")
            self.unprivileged_groups = unprivileged_groups
            self.privileged_groups = privileged_groups
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
            # Define lists to keep track of the fitting progress
            self.classifier_losses = []
            self.adversary_losses = []
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
        if type(m) != nn.Linear:
            pass # Skip modules without weights
        elif self.weights_method.lower() == "uniform":
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
                nn.init.zeros_(m.bias.data)
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
        
        # Set random seed
        if self.seed is not None:
            torch.manual_seed(self.seed)
        
        # Map the dataset labels to 0 and 1.
        temp_labels = dataset.labels.copy()
        temp_labels[(dataset.labels == dataset.favorable_label).ravel(),0] = 1.0
        temp_labels[(dataset.labels == dataset.unfavorable_label).ravel(),0] = 0.0
        
        # Define the number of samples, features, global training steps
        num_train_samples, features_dim = np.shape(dataset.features)
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
        
        # Instanciate the classifier and (if debias is set to True) adversary models
        if type(self.classifier_args) is not None:
            classifier_args = [self.classifier_args[0](features_dim, self.classifier_num_hidden_units,
                                                      self.classifier_dropout_ps) ] + self.classifier_args[1:]
        else:
            classifier_args = self.classifier_args
        classifier = self._classifier_model(*classifier_args).to(self.device)
        classifier = classifier.apply(self.weights_init)
        if self.debias:
            adversary = self._adversary_model().to(self.device) if type(self.adversary_args
            ) is None else self._adversary_model(*self.adversary_args).to(self.device)
            adversary = adversary.apply(self.weights_init)
        
        # Setup optimizers with exponentially decaying learning rate schedulers
        classifier_optim = self.optimizer([p for p in classifier.parameters() if p.requires_grad], lr=self.init_lr)
        classifier_lr_scheduler = StaircaseExponentialLR(classifier_optim, global_steps, self.init_lr, self.decay_steps,
                                                        self.decay_rate, self.lr_clip, self.staircase, self.verbose)
        classifier_lr_scheduler.initial_lr = self.init_lr
        if self.debias:
            adversary_optim = self.optimizer([p for p in adversary.parameters() if p.requires_grad], lr=self.init_lr)
            adversary_lr_scheduler = StaircaseExponentialLR(adversary_optim, global_steps, self.init_lr, self.decay_steps,
                                                           self.decay_rate, self.lr_clip, self.staircase, self.verbose)
            adversary_lr_scheduler.initial_lr = self.init_lr
        
        # Define the loss function criteria & tracking list for model(s)
        criterion = nn.BCELoss(reduction="mean")
        
        # Begin to train both models over each epoch
        global_step = 0
        if self.verbose:
            print(f'Starting to train model(s) on {self.device.upper()}:')
        classifier.train()
        if self.debias:
            adversary.train()
        for epoch in range(self.num_epochs):
            for i, data in enumerate(train_loader, 0):
                # Update learning rate(s)
                classifier_lr_scheduler.step(global_step, classifier.__class__.__name__)
                if self.debias:
                    adversary_lr_scheduler.step(global_step, adversary.__class__.__name__)
                # Train the classifier model
                classifier.zero_grad()
                batch_features = data[:][0].to(self.device)
                batch_labels = data[:][1].to(self.device)
                pred_labels, pred_logits = classifier(batch_features)
                classifier_error = criterion(pred_labels, batch_labels)
                classifier_error.backward()
                self.classifier_losses.append(classifier_error.item())
                classifier_mean_error = np.mean(self.classifier_losses)
                if self.debias: # Train the adversary model
                    adversary.zero_grad()
                    batch_protected_attributes = data[:][2].to(self.device)
                    pred_protected_attributes_labels, pred_protected_attributes_logits = adversary(
                    pred_logits, batch_labels)
                    adversary_error = criterion(pred_protected_attributes_labels, batch_protected_attributes)
                    adversary_error.backward()
                    self.adversary_losses.append(adversary_error.item())
                    adversary_mean_error = np.mean(self.adversary_losses)
                    adversary_optim.step() # Update adversary model parameters
                classifier_optim.step() # Update classifier model parameters
                # Print training statistics if verbose option is set to True
                if self.verbose and self.debias and i % 200 == 0:
                    print("Epoch: [%d/%d] Batch: [%d/%d]\tClassifier_Loss: %.4f\tAdversary Loss: %.4f\tC(x): %.4f\tA(x, y): %.4f" % \
                    (epoch + 1, self.num_epochs, i + 1, len(train_loader), classifier_error.item(),
                    adversary_error.item(), classifier_mean_error, adversary_mean_error))
                elif self.verbose and i % 200 == 0:
                    print("Epoch: [%d/%d] Batch: [%d/%d]\tClassifier Loss: %.4f\tC(x): %.4f" % \
                    (epoch + 1, self.num_epochs, i + 1, len(train_loader), classifier_error.item(), classifier_mean_error))
                # Save epoch losses for later plotting
                self.classifier_losses.append(classifier_error.item())
                if self.debias:
                    self.adversary_losses.append(adversary_error.item())
                global_step += 1
        classifier.training = False
        self._classifier_model = classifier
        if self.debias:
            adversary.training = False
            self._adversary_model = adversary
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
        
        if self._classifier_model.training:
            raise NotFittedError("The AdversarialDebiasing fit method must first be executed before using the predict method.")
        
        if self.seed is not None:
            torch.manual_seed(self.seed)

        features = torch.from_numpy(dataset.features).float().to(self.device)
        
        pred_labels = self._classifier_model.forward(features)[0]
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