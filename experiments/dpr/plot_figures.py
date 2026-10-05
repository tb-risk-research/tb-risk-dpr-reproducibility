"""Current publication snapshot exporter; historical renderer is in legacy/."""
import sys
from export_publication import main
sys.argv += ['--kind', 'figures']
if __name__=='__main__': main()
