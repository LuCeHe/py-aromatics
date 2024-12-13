from pyaromatics.keras_tools.esoteric_tasks.long_range_arena import lra_tasks

language_tasks = ['ptb', 'wiki103', 'wmt14', 'time_ae_merge', 'monkey', 'wordptb', 'wordptb1', ] + \
                 ['lra_' + t for t in lra_tasks]
