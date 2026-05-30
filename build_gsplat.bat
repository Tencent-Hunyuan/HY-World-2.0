@echo off
echo Building gsplat v1.5.3 from source...
C:\workspace\world\HY-World-2.0\.venv\Scripts\pip.exe install C:\workspace\world\gsplat --no-build-isolation < NUL
echo Done. Exit: %ERRORLEVEL%
