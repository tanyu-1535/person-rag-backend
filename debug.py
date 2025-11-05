import os, sys, traceback, uvicorn, app

port = int(os.environ["PORT"])
print("[DEBUG] PORT =", port, file=sys.stderr)
try:
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        log_level="debug",
    )
except Exception as e:
    print("[CRASH]", e, file=sys.stderr)
    traceback.print_exc()
    raise