param([string]$FileName = '*.docx')
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$sourceDir = Join-Path $taskRoot 'output\split_reports'
$renderDir = Join-Path $taskRoot 'tmp\docx_qa'
New-Item -ItemType Directory -Force -Path $renderDir | Out-Null
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    foreach ($file in Get-ChildItem -LiteralPath $sourceDir -Filter $FileName) {
        $document = $word.Documents.Open($file.FullName, $false, $true)
        try {
            $pdfPath = Join-Path $renderDir ($file.BaseName + '.pdf')
            $document.ExportAsFixedFormat($pdfPath, 17)
            Write-Output $pdfPath
        } finally {
            $document.Close(0)
            [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($document)
        }
    }
} finally {
    $word.Quit()
    [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($word)
}
