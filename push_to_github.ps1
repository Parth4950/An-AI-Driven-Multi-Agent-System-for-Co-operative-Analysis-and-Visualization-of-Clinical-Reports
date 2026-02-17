# Push clinical_analyzer to GitHub
# Run this in PowerShell from the project root (where this script lives).
# Requires: Git installed and GitHub auth (SSH key or credential manager).

Set-Location $PSScriptRoot

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Error "Git is not installed or not in PATH. Install Git and try again."
    exit 1
}

if (-not (Test-Path .git)) {
    git init
}

$remote = "https://github.com/Parth4950/An-AI-Driven-Multi-Agent-System-for-Co-operative-Analysis-and-Visualization-of-Clinical-Reports.git"
if (-not (git remote get-url origin 2>$null)) {
    git remote add origin $remote
} else {
    git remote set-url origin $remote
}

git add .
git status
Write-Host "`nReview the files above. .env is ignored and will NOT be committed.`n"
git commit -m "Scaffold: project structure, .env placeholder, requirements, .cursorrules"
git branch -M main
git push -u origin main

Write-Host "`nDone. If push failed, ensure you're authenticated (GitHub CLI, SSH, or credential manager)."
