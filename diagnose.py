#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagnostic script - captures full traceback on error"""
import sys
import os
import traceback

# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

log_file = os.path.join(script_dir, "diagnostic_log.txt")

def log(msg):
    print(msg)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

try:
    log("=" * 50)
    log(f"Python: {sys.executable}")
    log(f"Version: {sys.version}")
    log(f"sys.path[0:5]: {sys.path[:5]}")
    log(f"Working dir: {os.getcwd()}")
    log("=" * 50)
    log("")
    
    log("[1] Importing tb_risk...")
    import tb_risk
    log(f"    tb_risk version: {tb_risk.__version__}")
    log(f"    tb_risk path: {tb_risk.__file__}")
    
    log("[2] Importing TB_Risk_Assessment...")
    from tb_risk.assessment import TB_Risk_Assessment
    log("    Import OK")
    
    log("[3] Creating app instance...")
    app = TB_Risk_Assessment()
    log("    Instance created")
    
    log("[4] Initializing GUI...")
    app.init_gui()
    log("    GUI initialized")
    log(f"    Root window: {app.root}")
    log(f"    Root title: {app.root.title() if app.root else 'None'}")
    
    log("")
    log("[5] Starting mainloop (will close after 3 seconds for test)...")
    log("")
    log("=" * 50)
    log("If you see a window popup, the launch is working!")
    log("=" * 50)
    
    # Auto close after 3 seconds
    if app.root:
        app.root.after(3000, app.root.destroy)
        app.root.mainloop()
        log("")
        log("Mainloop exited normally.")
    else:
        log("ERROR: No root window!")
        
except Exception as e:
    log("")
    log("!" * 50)
    log("ERROR OCCURRED:")
    log("!" * 50)
    log(traceback.format_exc())
    log("")
    log(f"Error type: {type(e).__name__}")
    log(f"Error message: {e}")
    sys.exit(1)
