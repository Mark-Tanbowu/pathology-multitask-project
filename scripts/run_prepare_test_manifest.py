import subprocess
import sys

sys.exit(subprocess.call([sys.executable, "-m", "prepare.manifest_builder_test"] + sys.argv[1:]))
