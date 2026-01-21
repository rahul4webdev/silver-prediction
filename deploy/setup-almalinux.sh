#!/bin/bash
# Silver Price Prediction System - AlmaLinux 9 + CyberPanel Setup Script
# Run this script on your server after creating websites in CyberPanel

set -e

echo "=== Silver Price Prediction System Setup (AlmaLinux 9 + CyberPanel) ==="
echo ""

# Variables - Update these if needed
API_DOMAIN="predictionapi.gahfaudio.in"
FRONTEND_DOMAIN="prediction.gahfaudio.in"
API_PATH="/home/${API_DOMAIN}/public_html"
FRONTEND_PATH="/home/${FRONTEND_DOMAIN}/public_html"
DATA_PATH="/home/${API_DOMAIN}/data"
REPO_URL="https://github.com/rahul4webdev/silver-prediction.git"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    print_error "Please run as root (sudo ./setup-almalinux.sh)"
    exit 1
fi

echo "Step 1: Installing required packages..."
# Redis (if not already installed)
if ! command -v redis-server &> /dev/null; then
    dnf install -y redis
    systemctl enable redis
    systemctl start redis
    print_status "Redis installed and started"
else
    print_status "Redis already installed"
fi

# PM2 for Node.js process management
if ! command -v pm2 &> /dev/null; then
    npm install -g pm2
    print_status "PM2 installed"
else
    print_status "PM2 already installed"
fi

echo ""
echo "Step 2: Setting up Backend (API)..."

# Create data directories
mkdir -p ${DATA_PATH}/models
mkdir -p ${DATA_PATH}/raw
mkdir -p ${DATA_PATH}/processed
mkdir -p /home/${API_DOMAIN}/logs
chown -R ${API_DOMAIN}:${API_DOMAIN} ${DATA_PATH}
chown -R ${API_DOMAIN}:${API_DOMAIN} /home/${API_DOMAIN}/logs
print_status "Data directories created"

# Clone repository to API path if not exists
if [ ! -d "${API_PATH}/.git" ]; then
    print_warning "Cloning repository to ${API_PATH}..."
    cd ${API_PATH}
    rm -rf * .*  2>/dev/null || true
    git clone ${REPO_URL} .
    chown -R ${API_DOMAIN}:${API_DOMAIN} ${API_PATH}
    print_status "Repository cloned"
else
    print_status "Repository already exists"
fi

# Create Python virtual environment
cd ${API_PATH}
if [ ! -d "venv" ]; then
    python3.11 -m venv venv
    print_status "Python virtual environment created"
else
    print_status "Virtual environment already exists"
fi

# Install Python dependencies
source venv/bin/activate
pip install --upgrade pip
pip install -r backend/requirements.txt
deactivate
print_status "Python dependencies installed"

echo ""
echo "Step 3: Setting up Frontend..."

# Create frontend logs directory
mkdir -p /home/${FRONTEND_DOMAIN}/logs
chown -R ${FRONTEND_DOMAIN}:${FRONTEND_DOMAIN} /home/${FRONTEND_DOMAIN}/logs

# Clone repository to Frontend path if not exists
if [ ! -d "${FRONTEND_PATH}/.git" ]; then
    print_warning "Cloning repository to ${FRONTEND_PATH}..."
    cd ${FRONTEND_PATH}
    rm -rf * .* 2>/dev/null || true
    git clone ${REPO_URL} .
    chown -R ${FRONTEND_DOMAIN}:${FRONTEND_DOMAIN} ${FRONTEND_PATH}
    print_status "Repository cloned to frontend"
else
    print_status "Frontend repository already exists"
fi

# Install Node.js dependencies and build
cd ${FRONTEND_PATH}/frontend
npm ci
print_status "Node.js dependencies installed"

# Create .env.local for frontend
cat > .env.local << 'EOF'
NEXT_PUBLIC_API_URL=https://predictionapi.gahfaudio.in
NEXT_PUBLIC_WS_URL=wss://predictionapi.gahfaudio.in
NEXT_PUBLIC_APP_NAME=Silver Prediction
EOF
print_status "Frontend .env.local created"

# Build frontend
npm run build
print_status "Frontend built successfully"

echo ""
echo "Step 4: Setting up Supervisor configurations..."

# Copy supervisor configs
cp ${API_PATH}/deploy/supervisor/*.conf /etc/supervisord.d/
print_status "Supervisor configs copied"

# Reload supervisor
supervisorctl reread
supervisorctl update
print_status "Supervisor updated"

echo ""
echo "Step 5: Setting up PM2 for frontend..."

# Start frontend with PM2
cd ${FRONTEND_PATH}/frontend
su - ${FRONTEND_DOMAIN} -c "cd ${FRONTEND_PATH}/frontend && pm2 start npm --name 'silver-prediction-frontend' -- start"
su - ${FRONTEND_DOMAIN} -c "pm2 save"

# Setup PM2 to start on boot
pm2 startup systemd -u ${FRONTEND_DOMAIN} --hp /home/${FRONTEND_DOMAIN}
print_status "PM2 configured for frontend"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo ""
echo "1. Create the PostgreSQL database:"
echo "   Log into CyberPanel -> Databases -> Create Database"
echo "   - Database Name: silver_prediction"
echo "   - User: prediction_user"
echo "   - Save the password"
echo ""
echo "2. Create .env file for backend:"
echo "   cd ${API_PATH}"
echo "   cp .env.example .env"
echo "   nano .env"
echo "   # Fill in your database credentials and other settings"
echo ""
echo "3. Configure vHost in CyberPanel:"
echo ""
echo "   For API (${API_DOMAIN}):"
echo "   CyberPanel -> Websites -> ${API_DOMAIN} -> vHost Conf"
echo "   Copy contents from: ${API_PATH}/deploy/openlitespeed/vhconf-api.conf"
echo ""
echo "   For Frontend (${FRONTEND_DOMAIN}):"
echo "   CyberPanel -> Websites -> ${FRONTEND_DOMAIN} -> vHost Conf"
echo "   Copy contents from: ${API_PATH}/deploy/openlitespeed/vhconf-frontend.conf"
echo ""
echo "4. Start backend services:"
echo "   supervisorctl start silver-prediction-api"
echo "   supervisorctl start silver-prediction-worker"
echo "   supervisorctl start silver-prediction-scheduler"
echo ""
echo "5. Restart OpenLiteSpeed:"
echo "   systemctl restart lsws"
echo ""
echo "6. Verify services:"
echo "   supervisorctl status"
echo "   pm2 status"
echo "   curl https://${API_DOMAIN}/api/v1/health"
echo ""
print_status "Setup script completed!"
