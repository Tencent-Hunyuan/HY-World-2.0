# Build gsplat v1.5.3 CUDA extension into HY-World-2.0 venv

$pip    = "C:\workspace\world\HY-World-2.0\.venv\Scripts\pip.exe"
$gsplat = "C:\workspace\world\gsplat"
$vsPath = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

Write-Host "Setting up MSVC environment..."

# Add vswhere to PATH so vcvars64.bat can find it
$env:PATH += ";C:\Program Files (x86)\Microsoft Visual Studio\Installer"

# Import MSVC env vars into current PowerShell session
$envDump = cmd /c "`"$vsPath`" >nul 2>&1 && set" 2>&1
foreach ($line in $envDump) {
    if ($line -match "^([^=]+)=(.*)$") {
        [System.Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], "Process")
    }
}

Write-Host "CL: $(where.exe cl 2>$null)"
Write-Host "Building gsplat from $gsplat ..."

& $pip install 