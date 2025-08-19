#!/bin/bash

# Vercel Deployment Script for TDS Data Analyst Agent

echo "🚀 Preparing Vercel deployment..."

# Backup original files
echo "📦 Backing up original files..."
cp requirements.txt requirements-original.txt
cp app.py app-original.py

# Use lightweight requirements for Vercel
echo "📝 Using lightweight requirements..."
cp requirements-vercel.txt requirements.txt

# Use simplified app for Vercel
echo "🔄 Using simplified app..."
cp app-vercel.py app.py

echo "✅ Ready for Vercel deployment!"
echo ""
echo "Now run:"
echo "  vercel --prod"
echo ""
echo "After deployment, you can restore original files with:"
echo "  cp requirements-original.txt requirements.txt"
echo "  cp app-original.py app.py"
