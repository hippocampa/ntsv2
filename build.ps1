# Build script for production deployment

Write-Host "Building notesbyts academic site..." -ForegroundColor Green

# Clean previous build
if (Test-Path "public") {
    Remove-Item -Recurse -Force "public"
    Write-Host "Cleaned previous build" -ForegroundColor Yellow
}

# Build the site
Write-Host "Running Hugo build..." -ForegroundColor Blue
hugo --minify --gc

if ($LASTEXITCODE -eq 0) {
    Write-Host "Build completed successfully!" -ForegroundColor Green
    Write-Host "Files are ready in the 'public' directory" -ForegroundColor Cyan
    
    # Display build statistics
    $files = Get-ChildItem -Recurse "public" | Measure-Object
    Write-Host "Generated $($files.Count) files" -ForegroundColor Gray
    
    # Display size
    $size = Get-ChildItem -Recurse "public" | Measure-Object -Property Length -Sum
    $sizeInMB = [math]::Round($size.Sum / 1MB, 2)
    Write-Host "Total size: $sizeInMB MB" -ForegroundColor Gray
} else {
    Write-Host "Build failed!" -ForegroundColor Red
    exit 1
}
