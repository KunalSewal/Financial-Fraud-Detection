# Dashboard Setup Script
# Run this from the project root

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "🎨 Fraud Detection Dashboard Setup" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Check if Node.js is installed
Write-Host "Checking Node.js installation..." -ForegroundColor Yellow
$nodeVersion = node --version 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Node.js is not installed!" -ForegroundColor Red
    Write-Host "Please install Node.js 18+ from https://nodejs.org/" -ForegroundColor Yellow
    exit 1
}
Write-Host "✅ Node.js $nodeVersion found" -ForegroundColor Green

# Navigate to dashboard directory
Write-Host "`nNavigating to dashboard directory..." -ForegroundColor Yellow
Set-Location dashboard

# Check if package.json exists
if (-not (Test-Path "package.json")) {
    Write-Host "❌ package.json not found!" -ForegroundColor Red
    exit 1
}

# Install dependencies
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "📦 Installing dependencies..." -ForegroundColor Cyan
Write-Host "This may take 2-3 minutes..." -ForegroundColor Yellow
Write-Host "========================================`n" -ForegroundColor Cyan

npm install

if ($LASTEXITCODE -ne 0) {
    Write-Host "`n❌ Installation failed!" -ForegroundColor Red
    exit 1
}

Write-Host "`n✅ All dependencies installed successfully!" -ForegroundColor Green

# Create .env.local file
Write-Host "`nCreating environment configuration..." -ForegroundColor Yellow
$envContent = @"
# ML Backend API
NEXT_PUBLIC_API_URL=http://localhost:8000

# Weights & Biases (optional)
NEXT_PUBLIC_WANDB_ENTITY=your-username
NEXT_PUBLIC_WANDB_PROJECT=fraud-detection-phase1
"@

Set-Content -Path ".env.local" -Value $envContent
Write-Host "✅ Created .env.local file" -ForegroundColor Green

# Final instructions
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "🚀 Setup Complete!" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Start the development server:" -ForegroundColor White
Write-Host "     cd dashboard" -ForegroundColor Gray
Write-Host "     npm run dev" -ForegroundColor Gray
Write-Host ""
Write-Host "  2. Open your browser:" -ForegroundColor White
Write-Host "     http://localhost:3000" -ForegroundColor Gray
Write-Host ""
Write-Host "  3. (Optional) Start ML backend:" -ForegroundColor White
Write-Host "     cd api" -ForegroundColor Gray
Write-Host "     uvicorn main:app --reload" -ForegroundColor Gray
Write-Host ""

Write-Host "📚 Documentation:" -ForegroundColor Yellow
Write-Host "  - Dashboard README: dashboard/README.md" -ForegroundColor White
Write-Host "  - Main project: README.md" -ForegroundColor White
Write-Host ""

Write-Host "✨ Happy building!" -ForegroundColor Cyan
