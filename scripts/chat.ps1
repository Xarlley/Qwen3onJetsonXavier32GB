#Requires -Version 5.1
<#
.SYNOPSIS
  Jetson Xavier vLLM 连续多轮对话客户端（Windows PowerShell）

.DESCRIPTION
  对已在跑的 OpenAI 兼容服务 /v1/chat/completions 做多轮 chat。
  默认连 USB 网段 Jetson：http://192.168.100.2:8000/v1
  不修改本机代理/TUN 配置；不重启板端服务。

.PARAMETER BaseUrl
  API 根，须含 /v1。默认 http://192.168.100.2:8000/v1

.PARAMETER Model
  模型 id。省略则请求 GET /v1/models 取第一个。

.PARAMETER MaxTokens
  默认 512

.PARAMETER Temperature
  默认 0.7

.PARAMETER NoStream
  强制非流式

.PARAMETER ShowTiming
  显示 TTFT（流式）与整轮耗时

.EXAMPLE
  cd C:\path\to\this\repo\scripts
  powershell -ExecutionPolicy Bypass -File .\chat.ps1

.EXAMPLE
  .\chat.ps1 -BaseUrl http://192.168.100.2:8000/v1 -ShowTiming
#>
[CmdletBinding()]
param(
    [string]$BaseUrl = "http://192.168.100.2:8000/v1",
    [string]$Model = "",
    [int]$MaxTokens = 512,
    [double]$Temperature = 0.7,
    [switch]$NoStream,
    [switch]$ShowTiming
)

$ErrorActionPreference = "Stop"

# --- UTF-8 console for Chinese ---
try {
    if ($Host.UI.RawUI) {
        try { [Console]::InputEncoding = [System.Text.UTF8Encoding]::new($false) } catch {}
        try { [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false) } catch {}
    }
    $OutputEncoding = [System.Text.UTF8Encoding]::new($false)
    chcp 65001 | Out-Null
} catch {}

function Write-Info([string]$msg) { Write-Host $msg -ForegroundColor Cyan }
function Write-Warn2([string]$msg) { Write-Host $msg -ForegroundColor Yellow }
function Write-Err2([string]$msg) { Write-Host $msg -ForegroundColor Red }
function Write-Assist([string]$msg) { Write-Host $msg -ForegroundColor Green }

function Show-Help {
    Write-Host ""
    Write-Host "命令:"
    Write-Host "  exit / quit / 退出     结束对话"
    Write-Host "  clear / 清空           清空多轮历史"
    Write-Host "  help / 帮助            显示本帮助"
    Write-Host "  （空行忽略）"
    Write-Host ""
    Write-Host "参数（启动时）:"
    Write-Host "  -BaseUrl      默认 http://192.168.100.2:8000/v1"
    Write-Host "  -Model        省略则自动 /v1/models"
    Write-Host "  -MaxTokens    默认 512"
    Write-Host "  -Temperature  默认 0.7"
    Write-Host "  -NoStream     禁用流式"
    Write-Host "  -ShowTiming   显示 TTFT / 耗时"
    Write-Host ""
    Write-Host "说明: 本地 vLLM 无需 API Key。USB 网卡若无 192.168.100.x，请先 SSH 转发："
    Write-Host "      ssh -N -L 18000:127.0.0.1:8000 jetson"
    Write-Host "      .\chat.ps1 -BaseUrl http://127.0.0.1:18000/v1"
    Write-Host ""
}

function Get-Json($uri, $timeoutSec = 15) {
    $req = [System.Net.HttpWebRequest]::Create($uri)
    $req.Method = "GET"
    $req.Timeout = $timeoutSec * 1000
    $req.ReadWriteTimeout = $timeoutSec * 1000
    $req.Accept = "application/json"
    $resp = $req.GetResponse()
    $sr = New-Object System.IO.StreamReader($resp.GetResponseStream(), [System.Text.Encoding]::UTF8)
    $text = $sr.ReadToEnd()
    $sr.Close(); $resp.Close()
    return ($text | ConvertFrom-Json)
}

function Resolve-ModelId([string]$base, [string]$hint) {
    if ($hint -and $hint.Trim().Length -gt 0) { return $hint.Trim() }
    $m = Get-Json ("{0}/models" -f $base.TrimEnd("/"))
    if (-not $m.data -or $m.data.Count -lt 1) {
        throw "GET /v1/models 未返回模型。请确认 vLLM 已加载权重。"
    }
    return [string]$m.data[0].id
}

function ConvertTo-ChatBodyJson($model, $messages, $maxTokens, $temp, [bool]$stream) {
    # Use ConvertTo-Json on strings so escaping is correct; force messages to a JSON array (PS 5.1).
    $msgJsonParts = New-Object System.Collections.Generic.List[string]
    foreach ($m in @($messages)) {
        $roleJson = ([string]$m.role | ConvertTo-Json -Compress)
        $contentJson = ([string]$m.content | ConvertTo-Json -Compress)
        $msgJsonParts.Add(('{0}"role":{1},"content":{2}{3}' -f '{', $roleJson, $contentJson, '}'))
    }
    $msgsJson = '[' + [string]::Join(',', $msgJsonParts.ToArray()) + ']'
    $streamLit = if ($stream) { 'true' } else { 'false' }
    $tempLit = $temp.ToString([System.Globalization.CultureInfo]::InvariantCulture)
    $modelJson = ($model | ConvertTo-Json -Compress)
    return ('{0}"model":{1},"messages":{2},"max_tokens":{3},"temperature":{4},"stream":{5}{6}' -f '{', $modelJson, $msgsJson, $maxTokens, $tempLit, $streamLit, '}')
}

function Invoke-ChatNonStream($base, $model, $messages, $maxTokens, $temp) {
    $uri = "{0}/chat/completions" -f $base.TrimEnd("/")
    $json = ConvertTo-ChatBodyJson $model $messages $maxTokens $temp $false
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($json)

    $req = [System.Net.HttpWebRequest]::Create($uri)
    $req.Method = "POST"
    $req.ContentType = "application/json; charset=utf-8"
    $req.Accept = "application/json"
    $req.Timeout = 300000
    $req.ReadWriteTimeout = 300000
    $req.ContentLength = $bytes.Length
    $s = $req.GetRequestStream()
    $s.Write($bytes, 0, $bytes.Length)
    $s.Close()

    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $resp = $req.GetResponse()
    $sr = New-Object System.IO.StreamReader($resp.GetResponseStream(), [System.Text.Encoding]::UTF8)
    $text = $sr.ReadToEnd()
    $sr.Close(); $resp.Close()
    $sw.Stop()

    $obj = $text | ConvertFrom-Json
    $content = [string]$obj.choices[0].message.content
    return @{ Content = $content; ElapsedMs = $sw.ElapsedMilliseconds; TtftMs = $null }
}

function Invoke-ChatStream($base, $model, $messages, $maxTokens, $temp) {
    $uri = "{0}/chat/completions" -f $base.TrimEnd("/")
    $json = ConvertTo-ChatBodyJson $model $messages $maxTokens $temp $true
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($json)

    $req = [System.Net.HttpWebRequest]::Create($uri)
    $req.Method = "POST"
    $req.ContentType = "application/json; charset=utf-8"
    $req.Accept = "text/event-stream"
    $req.Timeout = 300000
    $req.ReadWriteTimeout = 300000
    $req.ContentLength = $bytes.Length
    $s = $req.GetRequestStream()
    $s.Write($bytes, 0, $bytes.Length)
    $s.Close()

    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $ttftMs = $null
    $sb = New-Object System.Text.StringBuilder

    $resp = $req.GetResponse()
    $stream = $resp.GetResponseStream()
    $sr = New-Object System.IO.StreamReader($stream, [System.Text.Encoding]::UTF8)

    Write-Host ""
    Write-Host -NoNewline "助手: " -ForegroundColor Green

    while ($null -ne ($line = $sr.ReadLine())) {
        if ([string]::IsNullOrWhiteSpace($line)) { continue }
        if (-not $line.StartsWith("data:")) { continue }
        $payload = $line.Substring(5).Trim()
        if ($payload -eq "[DONE]") { break }
        try {
            $chunk = $payload | ConvertFrom-Json
        } catch {
            continue
        }
        $delta = $null
        try { $delta = $chunk.choices[0].delta.content } catch {}
        if ($null -ne $delta -and "$delta".Length -gt 0) {
            if ($null -eq $ttftMs) { $ttftMs = $sw.ElapsedMilliseconds }
            [void]$sb.Append([string]$delta)
            Write-Host -NoNewline ([string]$delta)
        }
    }
    Write-Host ""
    $sr.Close(); $resp.Close()
    $sw.Stop()
    return @{ Content = $sb.ToString(); ElapsedMs = $sw.ElapsedMilliseconds; TtftMs = $ttftMs }
}

function Format-ConnError([Exception]$ex) {
    $msg = $ex.Message
    if ($ex.InnerException) { $msg = $msg + " | " + $ex.InnerException.Message }
    Write-Err2 ("请求失败: " + $msg)
    if ($msg -match "actively refused|无法连接|No connection|timed out|Timeout|NameResolution|无法解析") {
        Write-Warn2 "提示: 确认 USB 网线已连、Jetson 地址一般为 192.168.100.2、vLLM 在 8000 端口（~/start_vllm_qwen3.sh），可测 http://192.168.100.2:8000/health"
    }
}

[System.Net.ServicePointManager]::Expect100Continue = $false

$BaseUrl = $BaseUrl.TrimEnd("/")
Write-Info "=== Qwen3 / vLLM 连续对话 ==="
Write-Info ("BaseUrl: " + $BaseUrl)

try {
    $healthUri = ($BaseUrl -replace "/v1$", "") + "/health"
    $null = Get-Json $healthUri 10
    Write-Info ("健康检查: OK (" + $healthUri + ")")
} catch {
    Write-Warn2 ("健康检查未通过（仍可尝试对话）: " + $_.Exception.Message)
    Write-Warn2 "若超时/Connection refused："
    Write-Warn2 "  1) USB 网线 + ssh jetson 是否正常"
    Write-Warn2 "  2) 若 Windows 网卡不是 192.168.100.x（常见为 169.254.x.x），直连 192.168.100.2 会失败"
    Write-Warn2 "  3) 在本机另开窗口:  ssh -N -L 18000:127.0.0.1:8000 jetson"
    Write-Warn2 "     然后:  .\chat.ps1 -BaseUrl http://127.0.0.1:18000/v1"
}

try {
    $Model = Resolve-ModelId $BaseUrl $Model
} catch {
    Format-ConnError $_.Exception
    exit 1
}
Write-Info ("模型: " + $Model)
Write-Info ("max_tokens=" + $MaxTokens + " temperature=" + $Temperature + " stream=" + (-not $NoStream))
Show-Help

$history = New-Object System.Collections.Generic.List[object]

while ($true) {
    Write-Host ""
    Write-Host -NoNewline "你: " -ForegroundColor White
    $userText = Read-Host
    if ($null -eq $userText) { continue }
    $userText = $userText.Trim()
    if ($userText.Length -eq 0) { continue }

    $low = $userText.ToLowerInvariant()
    if ($low -eq "exit" -or $low -eq "quit" -or $userText -eq "退出") {
        Write-Info "再见。"
        break
    }
    if ($low -eq "clear" -or $userText -eq "清空") {
        $history.Clear()
        Write-Info "历史已清空。"
        continue
    }
    if ($low -eq "help" -or $userText -eq "帮助") {
        Show-Help
        continue
    }

    $history.Add(@{ role = "user"; content = $userText })

    $msgs = @()
    foreach ($m in $history) {
        $msgs += @{ role = [string]$m.role; content = [string]$m.content }
    }

    $result = $null
    try {
        if ($NoStream) {
            $result = Invoke-ChatNonStream $BaseUrl $Model $msgs $MaxTokens $Temperature
            Write-Host ""
            Write-Assist ("助手: " + $result.Content)
        } else {
            try {
                $result = Invoke-ChatStream $BaseUrl $Model $msgs $MaxTokens $Temperature
            } catch {
                Write-Warn2 ("流式失败，改用非流式: " + $_.Exception.Message)
                $result = Invoke-ChatNonStream $BaseUrl $Model $msgs $MaxTokens $Temperature
                Write-Host ""
                Write-Assist ("助手: " + $result.Content)
            }
        }
    } catch {
        Format-ConnError $_.Exception
        if ($history.Count -gt 0) { [void]$history.RemoveAt($history.Count - 1) }
        continue
    }

    if ($null -eq $result -or [string]::IsNullOrEmpty($result.Content)) {
        Write-Warn2 "（空回复）"
        if ($history.Count -gt 0) { [void]$history.RemoveAt($history.Count - 1) }
        continue
    }

    $history.Add(@{ role = "assistant"; content = $result.Content })

    if ($ShowTiming) {
        $ttft = if ($null -ne $result.TtftMs) { "{0} ms" -f $result.TtftMs } else { "n/a(非流式)" }
        Write-Host ("[耗时] TTFT={0}  E2E={1} ms  历史轮次约{2}" -f $ttft, $result.ElapsedMs, [math]::Floor($history.Count / 2)) -ForegroundColor DarkGray
    }
}
