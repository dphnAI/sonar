# Ruff formatter.
#
# Usage:
#    # Do work and commit your work.
#    # Format files that differ from origin/main.
#    .\formatting.ps1

# Stop on first error
$ErrorActionPreference = "Stop"

# Change to script directory and get root of git repo
Push-Location $PSScriptRoot
$ROOT = git rev-parse --show-toplevel
Set-Location $ROOT

# Get tool versions
$RUFF_VERSION = (ruff --version).Split(" ")[1]
$MYPY_VERSION = (mypy --version).Split(" ")[1]
$CODESPELL_VERSION = (codespell --version)
$ISORT_VERSION = (isort --vn)
$CLANGFORMAT_VERSION = (clang-format --version).Split(" ")[2]

# Version check function
function Test-ToolVersion {
    param($toolName, $currentVersion, $requiredVersion)
    if ($currentVersion -ne $requiredVersion) {
        Write-Error "Wrong $toolName version installed: $requiredVersion is required, not $currentVersion."
        exit 1
    }
}

# Get required versions from requirements file
$REQUIRED_RUFF = (Get-Content requirements/lint.txt | Select-String "ruff==").ToString().Split("==")[2]
$REQUIRED_ISORT = (Get-Content requirements/lint.txt | Select-String "isort").ToString().Split("==")[2]
$REQUIRED_CODESPELL = (Get-Content requirements/lint.txt | Select-String "codespell").ToString().Split("==")[2]
$REQUIRED_CLANGFORMAT = (Get-Content requirements/lint.txt | Select-String "clang-format").ToString().Split("==")[2]

# Check versions
Test-ToolVersion "ruff" $RUFF_VERSION $REQUIRED_RUFF
Test-ToolVersion "isort" $ISORT_VERSION $REQUIRED_ISORT
Test-ToolVersion "codespell" $CODESPELL_VERSION $REQUIRED_CODESPELL
Test-ToolVersion "clang-format" $CLANGFORMAT_VERSION $REQUIRED_CLANGFORMAT

# Codespell excludes
$CODESPELL_EXCLUDES = @('--skip', 'tests/benchmarks/sonnet.txt,build/**')

function Invoke-SpellCheck {
    param([string[]]$files)
    codespell $files
}

function Invoke-SpellCheckAll {
    codespell --toml pyproject.toml $CODESPELL_EXCLUDES
}

function Invoke-SpellCheckChanged {
    $MERGEBASE = git merge-base origin/main HEAD
    $changedFiles = git diff --name-only --diff-filter=ACM $MERGEBASE -- "*.py" "*.pyi"
    if ($changedFiles) {
        codespell $changedFiles $CODESPELL_EXCLUDES
    }
}

# Run Codespell based on arguments
if ($args[0] -eq '--files') {
    Invoke-SpellCheck $args[1..($args.Length-1)]
}
elseif ($args[0] -eq '--all') {
    Invoke-SpellCheckAll
}
else {
    Invoke-SpellCheckChanged
}
Write-Host 'Sonar codespell: Done'

# Ruff section
function Invoke-Lint {
    param([string[]]$files)
    ruff $files
}

function Invoke-LintChanged {
    $MERGEBASE = git merge-base origin/main HEAD
    $changedFiles = git diff --name-only --diff-filter=ACM $MERGEBASE -- "*.py" "*.pyi"
    if ($changedFiles) {
        ruff $changedFiles
    }
}

# Run Ruff based on arguments
if ($args[0] -eq '--files') {
    Invoke-Lint $args[1..($args.Length-1)]
}
elseif ($args[0] -eq '--all') {
    Invoke-Lint @("sonar", "tests")
}
else {
    Invoke-LintChanged
}
Write-Host 'Sonar ruff: Done'

# Isort section
function Invoke-IsortCheck {
    param([string[]]$files)
    isort $files
}

function Invoke-IsortCheckAll {
    isort .
}

function Invoke-IsortCheckChanged {
    $MERGEBASE = git merge-base origin/main HEAD
    $changedFiles = git diff --name-only --diff-filter=ACM $MERGEBASE -- "*.py" "*.pyi"
    if ($changedFiles) {
        isort $changedFiles
    }
}

# Run Isort based on arguments
if ($args[0] -eq '--files') {
    Invoke-IsortCheck $args[1..($args.Length-1)]
}
elseif ($args[0] -eq '--all') {
    Invoke-IsortCheckAll
}
else {
    Invoke-IsortCheckChanged
}
Write-Host 'Sonar isort: Done'

# Clang-format section
$CLANG_FORMAT_EXCLUDES = @(
    'kernels/moe/softmax.cu',
    'kernels/punica/bgmv/bgmv_bf16_bf16_bf16.cu',
    'kernels/punica/bgmv/bgmv_config.h',
    'kernels/punica/bgmv/bgmv_impl.cuh',
    'kernels/punica/bgmv/vec_dtypes.cuh',
    'kernels/punica/punica_ops.cu',
    'kernels/punica/type_convert.h',
    'kernels/quantization/gguf/ggml-common.h',
    'kernels/quantization/gguf/dequantize.cuh',
    'kernels/quantization/gguf/vecdotq.cuh',
    'kernels/quantization/gguf/mmq.cuh',
    'kernels/quantization/gguf/mmvq.cuh'
)

function Invoke-ClangFormat {
    param([string[]]$files)
    clang-format -i $files
}

function Invoke-ClangFormatChanged {
    $MERGEBASE = git merge-base origin/main HEAD
    $changedFiles = git diff --name-only --diff-filter=ACM $MERGEBASE -- "*.h" "*.cpp" "*.cu" "*.cuh" | 
        Where-Object { $file = $_; -not ($CLANG_FORMAT_EXCLUDES | Where-Object { $file -like "*$_*" }) }
    
    if ($changedFiles) {
        $changedFiles | ForEach-Object { clang-format -i $_ }
    }
}

function Invoke-ClangFormatAll {
    Get-ChildItem -Recurse -Path "kernels/" -Include @("*.h", "*.cpp", "*.cu", "*.cuh") |
        Where-Object { $file = $_.FullName; -not ($CLANG_FORMAT_EXCLUDES | Where-Object { $file -like "*$_*" }) } |
        ForEach-Object { clang-format -i $_.FullName }
}

# Run clang-format based on arguments
if ($args[0] -eq '--files') {
    Invoke-ClangFormat $args[1..($args.Length-1)]
}
elseif ($args[0] -eq '--all') {
    Invoke-ClangFormatAll
}
else {
    Invoke-ClangFormatChanged
}
Write-Host 'Sonar clang-format: Done'

# Check for unstaged changes
$hasChanges = git diff --quiet
if (-not $hasChanges) {
    Write-Host 'Reformatted files. Please review and stage the changes.'
    Write-Host 'Changes not staged for commit:'
    Write-Host
    git --no-pager diff --name-only
    exit 1
}

# Restore original location
Pop-Location