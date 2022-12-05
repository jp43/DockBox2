import os 
import sys

import re
import configparser
import copy

default_options = {'GENERAL': {'epochs': {'required': True, 'type': int},
                               'depth': {'default': 2, 'type': int},
                               'nrof_neigh': {'default': 25, 'type': int}},

'MINIBATCH': {'batch_size': {'default': 2, 'type': int},
              'num_parallel_calls': {'default': 1, 'type': int}},

'OPTIMIZER': {'initial_learning_rate': {'default': 1e-3, 'type': float},
              'decay_steps': {'default': 1000, 'type': int},
              'decay_rate': {'default': 0.99, 'type': float},
              'staircase': {'default': True, 'type': bool}},

'LOSSN': {'type': {'default': 'BinaryFocalCrossentropy', 'among': ['BinaryFocalCrossentropy', 'BinaryCrossEntropyLoss']},
          'alpha': {'default': 0.5, 'type': float, 'with': ('type', 'BinaryFocalCrossentropy')},
          'gamma': {'default': 2.0, 'type': float, 'with': ('type', 'BinaryFocalCrossentropy')},
          'weight': {'default': 1.0, 'type': float}},

'LOSSR': {'weight': {'default': 1.0, 'type': float}},

'AGGREGATOR': {'shape': {'required': True, 'type': 'shape'},
               'type': {'default': 'maxpool', 'among': ['maxpool', 'mean']},
               'use_concat': {'default': True, 'type': bool},
               'activation': {'default': 'relu'}},

'EDGE': {'type': {'default': None, 'among': ['rmsd', 'cog'], 'type': list},
         'depth': {'default': 1, 'type': int},
         'activation': {'default': 'relu'}},

'CLASSIFIER': {'shape': {'default': '1', 'type': 'shape'},
               'activation_h': {'default': 'relu'},
               'activation': {'default': 'sigmoid'}},
}

default_options['LOSSG'] = copy.deepcopy(default_options['LOSSN'])
default_options['LOSSG']['weight']['default'] = .0

class ConfigSetup(object):

    def __init__(self, inifile):

        self.inifile = inifile
        if not os.path.isfile(inifile):
            raise IOError("File %s does not exist!"%inifile)

        self.load_parameters(inifile)

    def load_parameters(self, inifile):

        config = configparser.SafeConfigParser()
        config.read(inifile)

        # check if required options have been set
        for section in default_options:
            for option, properties in default_options[section].items():

                if 'required' in properties and properties['required']:
                    if not section in config.sections():
                        sys.exit("section %s is mandatory in .ini configuration file!"%section)

                    if not config.has_option(section, option):
                        sys.exit("option %s in section %s is mandatory in .ini configuration file!"%(option, section))

        parameters = {}
        # store settings in parameters
        for section in config.sections():
            if section not in default_options:
                sys.exit("section %s not recognized in .ini configuration file!"%section)
            else:
                options = dict(config.items(section))
                for option in options:
                   if option not in default_options[section]:
                       sys.exit("option %s in section %s not recognized in .ini configuration file"%(option, section))

                   else:
                       if section not in parameters:
                           parameters[section] = {}

                       option_value = config.get(section, option)
                       options_settings = default_options[section][option]

                       if 'none' in option_value.lower():
                           option_value = None

                       # converting option
                       elif 'type' in options_settings and options_settings['type'] != 'shape':
                           if options_settings['type'] != bool:
                               converter = options_settings['type']

                               if options_settings['type'] == list:
                                   option_value = map(str.strip, re.sub(r'[()]', '', option_value).split(','))
                               option_value = converter(option_value)

                           elif option_value.lower() in ['true', 'yes', '1']:
                               option_value = True

                           elif option_value.lower() in ['false', 'no', '0']:
                               option_value = False
                           else:
                               raise ValueError("Option %s in section %s should be boolean!"%(option, section))


                       if option_value is not None and 'among' in options_settings:
                           if 'type' in options_settings and options_settings['type'] == list:
                               for item in option_value:
                                   if item not in options_settings['among']:
                                       known_values = ', '.join(list(map(str, options_settings['among'])))
                                       raise ValueError("Option %s in section %s should be among %s"%(option, section, known_values))

                           elif option_value not in options_settings['among']:
                               known_values = ', '.join(list(map(str, options_settings['among'])))
                               raise ValueError("Option %s in section %s should be among %s"%(option, section, known_values))
                       
                       parameters[section][option] = option_value

        # set default options if they have not been set
        for section in default_options:
            for option, properties in default_options[section].items():
                if 'default' in properties:
                    if section not in parameters:
                        parameters[section] = {}

                    if option not in parameters[section]:
                        parameters[section][option] = properties['default']

        # set shape type options
        for section in default_options:
            for option, properties in default_options[section].items():

                if 'type' in properties and default_options[section][option]['type'] == 'shape':
                    value = parameters[section][option]
                    if value is not None:
                        value = list(map(int, re.sub(r'[()]', '', value).split(',')))

                        if section == 'AGGREGATOR':
                            depth = parameters['GENERAL']['depth']
                            if len(value) == 1:
                                value = value*depth
                            elif len(value) != depth:
                                raise ValueError("Aggregator shapes should match depth option in GENERAL section!")
                        parameters[section][option] = value

        # remove unrelated options
        for section in default_options:
            for option, properties in default_options[section].items():
                if 'with' in properties:
                    related_option, value = properties['with']

                    if parameters[section][related_option] != value:
                        parameters[section].pop(option)

        # set general options as direct attributes
        self.epochs = parameters['GENERAL']['epochs']
        self.depth = parameters['GENERAL']['depth']

        self.nrof_neigh = parameters['GENERAL']['nrof_neigh']

        self.minibatch = parameters['MINIBATCH']
        self.loss = {'loss_n': parameters['LOSSN'],
            'loss_g': parameters['LOSSG'],
            'loss_reg': parameters['LOSSR']}

        self.general = parameters['GENERAL']
        self.optimizer = parameters['OPTIMIZER']
        self.aggregator = parameters['AGGREGATOR']

        self.classifier = parameters['CLASSIFIER']
        self.edge_options = parameters['EDGE']

    def pretty_print(self):

        print("I will use the following options:")
        for attribute in ['optimizer', 'minibatch', 'general', 'aggregator', 'edge_options', 'classifier', 'loss']:
            options = getattr(self, attribute)
            options_info = ""

            for key, value in options.items():
                options_info += str(key) + ': ' + str(value) + ', '
            print(attribute.upper()+':', options_info[:-2])
