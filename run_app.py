import os, sys
os.chdir("/Users/kieran/Projects/Investment")
sys.argv = ["streamlit", "run", "portfolio_app.py", "--server.port=8501", "--server.headless=true"]
from streamlit.web.cli import main
main()
