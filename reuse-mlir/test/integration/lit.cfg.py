import lit.formats

config.name = 'Reuse IR'
config.test_format = lit.formats.ShTest(True)

config.suffixes = ['.mlir']

config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.test_output_root, 'test')

config.substitutions.append((r'%reuse-opt',
    os.path.join(config.binary_path, 'reuse-opt')))

config.substitutions.append((r'%FileCheck', config.filecheck))
