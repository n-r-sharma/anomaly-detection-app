# Configure Git (first time only)

git config --global user.name "Your Name"
git config --global user.email "your@email.com"

# Go to your project

cd "/Users/nikashsharma/Desktop/T212 Project"

# Initialize Git

git init

# Create .gitignore

cat > .gitignore << 'EOF'
**pycache**/
\*.pyc
.DS_Store
venv/
.env
to_do.md
nohup.out
.streamlit/
EOF

# Stage all files

git add .

# Create first commit

git commit -m "Initial commit: Real-time anomaly detection app"

# Rename to main branch

git branch -M main
