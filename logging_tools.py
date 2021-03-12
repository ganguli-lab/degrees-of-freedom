import os
import sys
import logging
from time import localtime, strftime

def loggingSetup(logger, scriptname, dir_log, do_log=True, param_str = ''):
	"""
	Set up logging and name for associated artefacts.
	Logger must first exist by calling <logger = logging.getLogger("my logger")>
		in the main script.
	"""
	logger.setLevel(logging.INFO)

	## Set up artifact name
	if param_str == '':
		aname = 'artifact_%s_%s' % (scriptname, strftime('%Y-%m-%d-%H%M', localtime()))
	else:
		aname = 'artifact_%s_%s_%s' % (scriptname, param_str, strftime('%Y-%m-%d-%H%M', localtime()))

	## Set up the log to be stored in dir_log
	if not os.path.exists(dir_log):
		os.makedirs(dir_log)
	
	log_name = '%s_log.txt' % (aname)
	log_file = '%s/%s' % (dir_log, log_name)

	stream_handler = logging.StreamHandler()
	stream_handler.terminator = ''
	logger.addHandler(stream_handler)
	logger.propagate = False
	if do_log:
		file_handler = logging.FileHandler(log_file)
		file_handler.terminator = '' # This allows us better control over the newline
		logger.addHandler(file_handler)
		logger.info('Starting log with name %s\n',aname)
		logger.info('\n')
	
	return aname, do_log


def gitstatus(logger):
	"""
	[In progress]
	This function will print out the current commit and any diffs with master
	"""
	logger.info('==== Diffs ====\n')
	logger.info(os.popen('git diff --stat').read())
	logger.info('==== End Diffs ====\n')
	logger.info('\n')


def envstatus(logger, use_gpu=False):
	"""
	Prints the current environment and the CUDA version and driver to the log
	"""
	# Extract python version being run
	major = sys.version_info[0]
	minor = sys.version_info[1]
	logger.info('--- Running with Python%i.%i --- \n' % (major, minor))

	# Print out the packages
	packages = os.popen('python%i.%i -m pip list' % (major, minor)).read()
	logger.info('\n %s \n' % (packages)) 

	if use_gpu:
		logger.info('\n ---CUDA Details---\n')
		logger.info('\n')
		version = os.popen('cat /usr/local/cuda/version.txt').read()
		logger.info('%s \n' % (version)) 
		logger.info(os.popen('cat /proc/driver/nvidia/version').read())



def rnginit(str):
	"""
	[In progress]
	Need to read more about how to make random seed reproducible in python
	"""
	return


