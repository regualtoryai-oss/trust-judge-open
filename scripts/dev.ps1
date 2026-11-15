param([switch]$Stop)
$ErrorActionPreference = 'Stop'
$taskRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$taskRuntime = Join-Path $taskRoot '.runtime'
$taskRegistry = Join-Path $taskRuntime 'dev-processes.json'

function Stop-OwnedProcesses {
    if (-not (Test-Path -LiteralPath $taskRegistry)) { return }
    $taskRecords = Get-Content -LiteralPath $taskRegistry -Raw | ConvertFrom-Json
    foreach ($taskRecord in $taskRecords) {
        $taskProcess = Get-CimInstance Win32_Process -Filter "ProcessId=$($taskRecord.pid)" -ErrorAction SilentlyContinue
        # A recycled PID must never stop another application.
        if ($taskProcess -and $taskProcess.CommandLine -and
            $taskRecord.owner_root -eq $taskRoot -and
            $taskProcess.CommandLine -eq $taskRecord.command_line -and
            $taskProcess.CreationDate.ToUniversalTime().Ticks -eq $taskRecord.created_ticks) {
            Stop-Process -Id $taskRecord.pid -ErrorAction SilentlyContinue
            Write-Host "Stopped $($taskRecord.name) (PID $($taskRecord.pid))"
        }
    }
    Remove-Item -LiteralPath $taskRegistry
}

if ($Stop) { Stop-OwnedProcesses; exit }
if (Test-Path -LiteralPath $taskRegistry) {
    throw '已有开发进程记录，请先运行 .\scripts\dev.ps1 -Stop。'
}
$taskPython = Join-Path $taskRoot '.venv\Scripts\python.exe'
$taskVite = Join-Path $taskRoot 'web\node_modules\vite\bin\vite.js'
if (-not (Test-Path -LiteralPath $taskPython) -or -not (Test-Path -LiteralPath $taskVite)) {
    throw '请先在项目根目录运行 uv sync，并在 web 目录运行 npm ci。'
}
foreach ($taskPort in @(8790,5190)) {
    if (Get-NetTCPConnection -LocalPort $taskPort -State Listen -ErrorAction SilentlyContinue) {
        throw "端口 $taskPort 已被占用，请复用现有服务或先停止它。"
    }
}
$taskNode = (Get-Command node.exe).Source
New-Item -ItemType Directory -Force -Path $taskRuntime | Out-Null
$taskSpecs = @(
    @{name='api'; file=$taskPython; args=@('-m','app.cli','serve','--port','8790'); cwd=$taskRoot},
    @{name='worker'; file=$taskPython; args=@('-m','app.cli','worker'); cwd=$taskRoot},
    @{name='web'; file=$taskNode; args=@(('"'+$taskVite+'"'),'--host','127.0.0.1','--port','5190'); cwd=(Join-Path $taskRoot 'web')}
)
$taskStarted = @()
function Save-ProcessTree {
    $taskAllProcesses = @(Get-CimInstance Win32_Process)
    $taskKnownIds = @($taskStarted | ForEach-Object { $_.pid })
    $taskAdded = $true
    while ($taskAdded) {
        $taskAdded = $false
        foreach ($taskInfo in $taskAllProcesses) {
            if ($taskInfo.ParentProcessId -in $taskKnownIds -and $taskInfo.ProcessId -notin $taskKnownIds) {
                $script:taskStarted += @{name='child'; pid=$taskInfo.ProcessId; created_ticks=$taskInfo.CreationDate.ToUniversalTime().Ticks; command_line=$taskInfo.CommandLine; owner_root=$taskRoot}
                $taskKnownIds += $taskInfo.ProcessId
                $taskAdded = $true
            }
        }
    }
    # Stop children before their redirector/launcher parents on Windows.
    @($taskStarted | Sort-Object @{Expression={if ($_.name -eq 'child') {0} else {1}}}) | ConvertTo-Json | Set-Content -LiteralPath $taskRegistry -Encoding UTF8
}
try {
    foreach ($taskSpec in $taskSpecs) {
        $taskChild = Start-Process -FilePath $taskSpec.file -ArgumentList $taskSpec.args `
            -WorkingDirectory $taskSpec.cwd -WindowStyle Hidden -PassThru `
            -RedirectStandardOutput (Join-Path $taskRuntime ($taskSpec.name+'.stdout.log')) `
            -RedirectStandardError (Join-Path $taskRuntime ($taskSpec.name+'.stderr.log'))
        $taskInfo = Get-CimInstance Win32_Process -Filter "ProcessId=$($taskChild.Id)"
        $taskStarted += @{name=$taskSpec.name; pid=$taskChild.Id; created_ticks=$taskInfo.CreationDate.ToUniversalTime().Ticks; command_line=$taskInfo.CommandLine; owner_root=$taskRoot}
        $taskStarted | ConvertTo-Json | Set-Content -LiteralPath $taskRegistry -Encoding UTF8
    }
    Start-Sleep -Seconds 2
    Save-ProcessTree
    foreach ($taskRecord in $taskStarted) {
        if (-not (Get-Process -Id $taskRecord.pid -ErrorAction SilentlyContinue)) {
            throw "$($taskRecord.name) 启动失败，请检查 .runtime 下对应日志。"
        }
    }
    Write-Host '工作台：http://127.0.0.1:5190'
    Write-Host 'API 文档：http://127.0.0.1:8790/docs'
    Write-Host '停止：.\scripts\dev.ps1 -Stop'
} catch {
    if ($taskStarted.Count) { Save-ProcessTree }
    Stop-OwnedProcesses
    throw
}
