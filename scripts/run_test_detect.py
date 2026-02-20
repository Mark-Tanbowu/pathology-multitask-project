import subprocess
import sys

sys.exit(subprocess.call([sys.executable, "-m", "src.engine.test_slide_detect"] + sys.argv[1:]))
