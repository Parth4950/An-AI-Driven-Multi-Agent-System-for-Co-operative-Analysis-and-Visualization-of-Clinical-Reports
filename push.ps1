# Push to GitHub using full path to Git (use when 'git' is not in PATH)
$git = "C:\Program Files\Git\bin\git.exe"
Set-Location $PSScriptRoot
# Remove stale lock files from interrupted git runs
Remove-Item -Path ".git\index.lock" -ErrorAction SilentlyContinue
Remove-Item -Path ".git\config.lock" -ErrorAction SilentlyContinue
& $git add .
& $git status
& $git commit -m "Clinical extraction pipeline: diabetes and blood pressure"
& $git branch -M main
& $git push -u origin main
