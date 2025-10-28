# Instala as dependências usadas por src/main.py, src/baseline.py e src/change_parameters.py
# Pacotes: numpy, scipy, scikit-learn, pandas, importlib_resources (backport para Python < 3.9)
# Uso (PowerShell):
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#   ./install_deps.ps1

$ErrorActionPreference = 'Stop'

Write-Host '==> Detectando Python/pip...' -ForegroundColor Cyan

function Get-PipInvocation {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        return @{ base='py'; useModule=$true }
    }
    elseif (Get-Command python -ErrorAction SilentlyContinue) {
        return @{ base='python'; useModule=$true }
    }
    elseif (Get-Command pip -ErrorAction SilentlyContinue) {
        return @{ base='pip'; useModule=$false }
    }
    else {
        throw 'Python/pip não encontrados no PATH. Instale Python 3.8+ e tente novamente.'
    }
}

$pip = Get-PipInvocation
$packages = @(
    'numpy>=1.21.5',
    'scipy>=1.5.2',
    'scikit-learn>=1.2.1',
    'pandas',
    'importlib_resources'
)

Write-Host '==> Instalando dependências com pip...' -ForegroundColor Cyan

try {
    if ($pip.useModule) {
        & $pip.base -m pip install --upgrade @packages
    }
    else {
        & $pip.base install --upgrade @packages
    }
    Write-Host '==> Instalação concluída com sucesso.' -ForegroundColor Green
}
catch {
    Write-Error "Falha ao instalar dependências: $($_.Exception.Message)"
    exit 1
}
