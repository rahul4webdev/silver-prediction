// PM2 configuration for Next.js frontend
module.exports = {
  apps: [
    {
      name: 'silver-prediction-frontend',
      cwd: '/var/www/silver-prediction/frontend',
      script: 'node_modules/next/dist/bin/next',
      args: 'start',
      instances: 1,
      exec_mode: 'fork',
      watch: false,
      max_memory_restart: '500M',
      env: {
        NODE_ENV: 'production',
        PORT: 3000,
      },
      error_file: '/var/log/pm2/silver-prediction-frontend-error.log',
      out_file: '/var/log/pm2/silver-prediction-frontend-out.log',
      log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
      merge_logs: true,
    },
  ],
};
