# Development server script

Write-Host "Starting Hugo development server..." -ForegroundColor Green
Write-Host "Your site will be available at: http://localhost:1313" -ForegroundColor Cyan

# Start Hugo server with drafts enabled
hugo server -D --disableFastRender --navigateToChanged
