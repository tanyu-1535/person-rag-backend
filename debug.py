# debug.py
import os, sys, traceback

try:
    port = int(os.environ['PORT'])
    print('[DEBUG] PORT =', port, file=sys.stderr, flush=True)

    # 先不启动 uvicorn，只验证能否导入
    import uvicorn
    print('[DEBUG] uvicorn import ok', file=sys.stderr, flush=True)

    import app
    print('[DEBUG] app import ok', file=sys.stderr, flush=True)

    # 再试着监听
    uvicorn.run(
    "app:app",
    host="0.0.0.0",      # 强制 IPv4
    port=port,
    log_level="debug",
    loop="asyncio",      # 排除 uvloop 对 IPv6 的默认偏好
)
except Exception as e:
    # 把异常同时打到 stderr 和本地文件，双保险
    traceback.print_exc()
    with open('/tmp/crash.log', 'w') as f:
        traceback.print_exc(file=f)
    raise

