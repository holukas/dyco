<#
.SYNOPSIS
    Build the dyco documentation and open it in a browser.

.DESCRIPTION
    Sphinx and its extensions are deliberately kept out of the project
    environment (see docs/requirements.txt), so this script runs them through
    `uv run --with-requirements`, which layers them into a temporary overlay and
    never touches uv.lock.

    Without -Watch the docs are built once and the start page is opened. With
    -Watch, sphinx-autobuild serves them on http://localhost:<Port> and rebuilds
    on every save.

.PARAMETER Page
    Which page to open, with or without the .html extension. Defaults to the
    start page.

.PARAMETER Watch
    Serve with live reload instead of building once.

.PARAMETER Strict
    Turn warnings into errors, matching the Read the Docs build
    (fail_on_warning: true in .readthedocs.yaml).

.PARAMETER Clean
    Discard the cached build environment and rebuild every page. Use this when a
    build reports "no targets are out of date" after you edited a source file.

.PARAMETER Port
    Port for -Watch. Default 8000.

.EXAMPLE
    .\preview_docs.ps1
    Build once, open the start page.

.EXAMPLE
    .\preview_docs.ps1 -Page example-irga-20hz
    Build once, open that page.

.EXAMPLE
    .\preview_docs.ps1 -Watch -Page example-qcl-10hz
    Serve with live reload, opening that page.

.EXAMPLE
    .\preview_docs.ps1 -Strict -Clean
    Full rebuild that fails on the first warning, as Read the Docs does.
#>

[CmdletBinding()]
param(
    [string]$Page = 'index',
    [switch]$Watch,
    [switch]$Strict,
    [switch]$Clean,
    [int]$Port = 8000
)

$ErrorActionPreference = 'Stop'

$repoRoot = $PSScriptRoot
$sourceDir = Join-Path $repoRoot 'docs'
$buildDir = Join-Path $repoRoot 'docs\_build\html'
$reqFile = Join-Path $sourceDir 'requirements.txt'

if (-not (Test-Path $reqFile)) {
    throw "Not found: $reqFile. Run this script from inside the dyco repository."
}
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv is not on PATH. See https://docs.astral.sh/uv/ for installation."
}

# Accept "example-irga-20hz", "example-irga-20hz.html" and "docs/example-irga-20hz.md" alike.
$pageName = [System.IO.Path]::GetFileNameWithoutExtension($Page)
$pageFile = Join-Path $buildDir "$pageName.html"
$sourceFile = Join-Path $sourceDir "$pageName.md"
if (-not (Test-Path $sourceFile)) {
    Write-Warning "No source page docs\$pageName.md. Building anyway; the browser may show a 404."
}

# Shared by both modes. -W turns warnings into errors, -E discards the cached
# environment, -a rewrites every output page (needed because -E alone still
# skips pages Sphinx believes are current).
$sphinxFlags = @()
if ($Strict) { $sphinxFlags += '-W' }
if ($Clean) { $sphinxFlags += @('-E', '-a') }

if ($Watch) {
    $openUrl = "http://localhost:$Port/$pageName.html"
    Write-Host "Serving docs on $openUrl (Ctrl+C to stop)" -ForegroundColor Cyan
    Write-Host "Rebuilds on every save." -ForegroundColor DarkGray

    # sphinx-autobuild opens the browser itself once the first build is served.
    $uvArgs = @(
        'run', '--with-requirements', $reqFile, '--with', 'sphinx-autobuild',
        'sphinx-autobuild', $sourceDir, $buildDir,
        '--port', $Port, '--open-browser', '--delay', '1'
    ) + $sphinxFlags
    & uv @uvArgs
    exit $LASTEXITCODE
}

Write-Host "Building docs -> $buildDir" -ForegroundColor Cyan
$uvArgs = @(
    'run', '--with-requirements', $reqFile,
    'sphinx-build', '-b', 'html'
) + $sphinxFlags + @($sourceDir, $buildDir)
& uv @uvArgs

if ($LASTEXITCODE -ne 0) {
    Write-Host ''
    Write-Host "Build failed (exit $LASTEXITCODE). Nothing opened." -ForegroundColor Red
    if ($Strict) {
        Write-Host "-Strict makes every warning fatal; rerun without it to see the page anyway." -ForegroundColor DarkGray
    }
    exit $LASTEXITCODE
}

if (-not (Test-Path $pageFile)) {
    throw "Build reported success but $pageFile is missing. Try -Clean."
}

Write-Host "Opening $pageFile" -ForegroundColor Green
Start-Process $pageFile
