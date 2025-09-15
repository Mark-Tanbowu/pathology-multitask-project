import subprocess
import sys

sys.exit(subprocess.call([sys.executable, "-m", "src.engine.infer"] + sys.argv[1:]))
