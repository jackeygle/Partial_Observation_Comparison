import sys
from configargparse import ArgumentParser, ArgumentDefaultsHelpFormatter  # better than argparse

parser = ArgumentParser(
	formatter_class=ArgumentDefaultsHelpFormatter,
	auto_env_var_prefix='PEDPRED_',
	default_config_files=['base.cfg','machine.cfg'],
	args_for_setting_config_path=('-c','--config-file'),
	args_for_writing_out_config_file=('-w','--write-config-file'),
)

g = parser.add_argument_group("Dataset")
g.add_argument('-d','--dataset',            default='atc:corridor',     help="dataset name, and specific grid configuration",   metavar='DATASET[:GRID]')
g.add_argument('--resolution',  type=float, default=1.0,    help="discretisation (spatial) resolution, in meters",              metavar='RES')
g.add_argument('--period',      type=float, default=1.0,    help="discretisation temporal resolution, in seconds",              metavar='D')
g.add_argument('--kernel',                  default='tri',  help="discretisation kernel shape", choices=('rect','tri','hann'),  metavar='K')

g = parser.add_argument_group("Training")
g.add_argument('-b','--batch',  type=int,   default=50,      help="number of training instances per iteration",                  metavar='N')
g.add_argument('--nin',         type=int,   default=1,     help="number of input steps to observe",                            metavar='N')
g.add_argument('--nout',        type=int,   default=1,     help="number of output steps to predict",                           metavar='N')
g.add_argument('--loss',                    default='mean total weighted NLLL',     help="loss metric used to train the model", metavar='METRIC')

cfg, sys.argv[1:] = parser.parse_known_args()


if __name__=='__main__':
	print(cfg)
	parser.print_values()
	print(f"Remaining args: {sys.argv[1:]}")
