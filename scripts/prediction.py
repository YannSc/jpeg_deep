from os import getcwd
from os.path import join

import argparse
import sys
sys.path.append(getcwd())

parser = argparse.ArgumentParser("Runs a prediction pass on a trained network.")
parser.add_argument("experiment", help="The experiment directory.")
parser.add_argument("weights", help="The weights to load.")
parser.add_argument('-gt', '--groundTruth', action='store_true', help="If the ground truth should be displayed.")
args = parser.parse_args()

# We reload the config to get everything as in the experiment
sys.path.append(join(args.experiment, "config"))
from config.pascalvoc.ssd.dct.dct_07.config_file import TrainingConfiguration as TC_DCT
config_dct = TC_DCT()

from config.pascalvoc.ssd.rgb.rgb_07.config_file import TrainingConfiguration as TC_RGB
config_rgb = TC_RGB()

def prep_model(config):

# Loading the model, for prepare for inference if required
    config.prepare_for_inference()
    config.network.load_weights(args.weights)
    model = config.network
    model.compile(loss=config.loss,
                  optimizer=config.optimizer,
                  metrics=config.metrics)

    config.prepare_testing_generator()
    config.test_generator.shuffle = True
    return model

model_dct = prep_model(config_dct)
# Getting the batch to process
X, _ = config_dct.test_generator.__getitem__(0)
# If the input is not a displayable stuff, get the displayable
X_true, y = config_dct.test_generator.get_raw_input_label(0)

print("Start DCT inference test")
i = 0
while (i<2000):

    y_pred = model_dct.predict(X)


model_rgb = prep_model(config_rgb)
X, _ = config_dct.test_generator.__getitem__(0)
# If the input is not a displayable stuff, get the displayable
X_true, y = config_dct.test_generator.get_raw_input_label(0)

print("Start RGB inference test")
i = 0
while (i<2000):

    y_pred = model_dct.predict(X)
