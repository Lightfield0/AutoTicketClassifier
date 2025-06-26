"""
🚀 Production Deployment Configuration
Production ortamı için deployment konfigürasyonları ve araçları
"""

import os
import json
import yaml
from pathlib import Path
from datetime import datetime

class DeploymentManager:
    """Production deployment yönetimi"""
    
    def __init__(self, project_root="."):
        self.project_root = Path(project_root)
        self.deployment_dir = self.project_root / "deployment"
        self.deployment_dir.mkdir(exist_ok=True)
        
    def generate_docker_files(self):
        """Docker deployment dosyalarını oluştur"""
        print("🐳 Docker dosyaları oluşturuluyor...")
        
        # Dockerfile
        dockerfile_content = '''# AutoTicket Classifier Production Dockerfile
FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/logs /app/models/trained /app/data /app/monitoring

# Set environment variables
ENV PYTHONPATH=/app
ENV FLASK_APP=web/app.py
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD python -c "from utils.monitoring import get_production_monitor; monitor = get_production_monitor(); print(monitor.health_check())" || exit 1

# Run application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--timeout", "120", "web.app:app"]
'''
        
        dockerfile_path = self.deployment_dir / "Dockerfile"
        with open(dockerfile_path, 'w') as f:
            f.write(dockerfile_content)
        
        # Docker Compose
        compose_config = {
            'version': '3.8',
            'services': {
                'autoticket-classifier': {
                    'build': {
                        'context': '..',
                        'dockerfile': 'deployment/Dockerfile'
                    },
                    'ports': ['5000:5000'],
                    'environment': [
                        'FLASK_ENV=production',
                        'PYTHONUNBUFFERED=1'
                    ],
                    'volumes': [
                        '../models/trained:/app/models/trained:ro',
                        '../logs:/app/logs',
                        '../data:/app/data:ro',
                        '../monitoring:/app/monitoring'
                    ],
                    'restart': 'unless-stopped',
                    'healthcheck': {
                        'test': ['CMD', 'python', '-c', 
                                "from utils.monitoring import get_production_monitor; print(get_production_monitor().health_check())"],
                        'interval': '30s',
                        'timeout': '10s',
                        'retries': 3,
                        'start_period': '30s'
                    }
                },
                'redis': {
                    'image': 'redis:7-alpine',
                    'ports': ['6379:6379'],
                    'restart': 'unless-stopped',
                    'command': 'redis-server --appendonly yes',
                    'volumes': ['redis_data:/data']
                }
            },
            'volumes': {
                'redis_data': {}
            }
        }
        
        compose_path = self.deployment_dir / "docker-compose.yml"
        with open(compose_path, 'w') as f:
            yaml.dump(compose_config, f, default_flow_style=False)
        
        print(f"✅ Docker dosyaları oluşturuldu: {self.deployment_dir}")
        return [dockerfile_path, compose_path]
    
    def generate_kubernetes_manifests(self):
        """Kubernetes deployment manifests"""
        print("☸️ Kubernetes manifests oluşturuluyor...")
        
        # Deployment manifest
        deployment_manifest = {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {
                'name': 'autoticket-classifier',
                'labels': {'app': 'autoticket-classifier'}
            },
            'spec': {
                'replicas': 3,
                'selector': {'matchLabels': {'app': 'autoticket-classifier'}},
                'template': {
                    'metadata': {'labels': {'app': 'autoticket-classifier'}},
                    'spec': {
                        'containers': [{
                            'name': 'autoticket-classifier',
                            'image': 'autoticket-classifier:latest',
                            'ports': [{'containerPort': 5000}],
                            'env': [
                                {'name': 'FLASK_ENV', 'value': 'production'},
                                {'name': 'PYTHONUNBUFFERED', 'value': '1'}
                            ],
                            'resources': {
                                'requests': {'memory': '512Mi', 'cpu': '250m'},
                                'limits': {'memory': '1Gi', 'cpu': '500m'}
                            },
                            'livenessProbe': {
                                'httpGet': {'path': '/health', 'port': 5000},
                                'initialDelaySeconds': 30,
                                'periodSeconds': 10
                            },
                            'readinessProbe': {
                                'httpGet': {'path': '/ready', 'port': 5000},
                                'initialDelaySeconds': 5,
                                'periodSeconds': 5
                            }
                        }]
                    }
                }
            }
        }
        
        # Service manifest
        service_manifest = {
            'apiVersion': 'v1',
            'kind': 'Service',
            'metadata': {
                'name': 'autoticket-classifier-service',
                'labels': {'app': 'autoticket-classifier'}
            },
            'spec': {
                'selector': {'app': 'autoticket-classifier'},
                'ports': [{'port': 80, 'targetPort': 5000}],
                'type': 'LoadBalancer'
            }
        }
        
        # Save manifests
        k8s_dir = self.deployment_dir / "kubernetes"
        k8s_dir.mkdir(exist_ok=True)
        
        with open(k8s_dir / "deployment.yaml", 'w') as f:
            yaml.dump(deployment_manifest, f, default_flow_style=False)
        
        with open(k8s_dir / "service.yaml", 'w') as f:
            yaml.dump(service_manifest, f, default_flow_style=False)
        
        print(f"✅ Kubernetes manifests oluşturuldu: {k8s_dir}")
        return k8s_dir
    
    def generate_production_requirements(self):
        """Production requirements.txt"""
        production_packages = [
            "streamlit>=1.28.0",
            "pandas>=1.5.0",
            "numpy>=1.24.0",
            "scikit-learn>=1.3.0",
            "matplotlib>=3.6.0",
            "seaborn>=0.12.0",
            "plotly>=5.15.0",
            "joblib>=1.3.0",
            "scipy>=1.10.0",
            "flask>=2.3.0",
            "gunicorn>=21.2.0",
            "redis>=4.5.0",
            "psutil>=5.9.0",
            "pyyaml>=6.0",
            "python-dotenv>=1.0.0",
            "prometheus-client>=0.17.0",
            "structlog>=23.1.0"
        ]
        
        req_path = self.deployment_dir / "requirements-production.txt"
        with open(req_path, 'w') as f:
            f.write('\\n'.join(production_packages))
        
        print(f"✅ Production requirements oluşturuldu: {req_path}")
        return req_path
    
    def generate_env_template(self):
        """Environment variables template"""
        env_template = '''# AutoTicket Classifier Environment Variables

# Flask Configuration
FLASK_APP=web/app.py
FLASK_ENV=production
SECRET_KEY=your-secret-key-here

# Database Configuration
DATABASE_URL=sqlite:///monitoring/production.db

# Redis Configuration
REDIS_URL=redis://localhost:6379/0

# Model Configuration
MODEL_PATH=models/trained/
DEFAULT_MODEL=ensemble

# Monitoring Configuration
MONITORING_ENABLED=true
LOG_LEVEL=INFO
METRICS_RETENTION_DAYS=30

# Performance Configuration
MAX_WORKERS=4
WORKER_TIMEOUT=120
MAX_REQUESTS_PER_WORKER=1000

# Security Configuration
ALLOWED_HOSTS=localhost,127.0.0.1
CORS_ORIGINS=http://localhost:3000

# Feature Flags
ENABLE_AB_TESTING=true
ENABLE_DRIFT_DETECTION=true
ENABLE_PERFORMANCE_MONITORING=true
'''
        
        env_path = self.deployment_dir / ".env.template"
        with open(env_path, 'w') as f:
            f.write(env_template)
        
        print(f"✅ Environment template oluşturuldu: {env_path}")
        return env_path
    
    def generate_deployment_scripts(self):
        """Deployment scripts"""
        scripts_dir = self.deployment_dir / "scripts"
        scripts_dir.mkdir(exist_ok=True)
        
        # Deploy script
        deploy_script = '''#!/bin/bash
set -e

echo "🚀 AutoTicket Classifier Deployment Starting..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Build Docker image
echo "📦 Building Docker image..."
docker build -t autoticket-classifier:latest -f deployment/Dockerfile .

# Stop existing containers
echo "🛑 Stopping existing containers..."
docker-compose -f deployment/docker-compose.yml down

# Start new containers
echo "🚀 Starting new containers..."
docker-compose -f deployment/docker-compose.yml up -d

# Wait for health check
echo "🏥 Waiting for health check..."
sleep 30

# Check if service is healthy
if curl -f http://localhost:5000/health > /dev/null 2>&1; then
    echo "✅ Deployment successful! Service is healthy."
else
    echo "❌ Deployment failed! Service is not responding."
    exit 1
fi

echo "🎉 Deployment completed successfully!"
'''
        
        deploy_path = scripts_dir / "deploy.sh"
        with open(deploy_path, 'w') as f:
            f.write(deploy_script)
        
        # Make executable
        deploy_path.chmod(0o755)
        
        # Rollback script
        rollback_script = '''#!/bin/bash
set -e

echo "🔄 Rolling back deployment..."

# Stop current containers
docker-compose -f deployment/docker-compose.yml down

# Start previous version (assuming previous:latest tag exists)
docker run -d -p 5000:5000 --name autoticket-classifier-rollback autoticket-classifier:previous

echo "✅ Rollback completed!"
'''
        
        rollback_path = scripts_dir / "rollback.sh"
        with open(rollback_path, 'w') as f:
            f.write(rollback_script)
        
        rollback_path.chmod(0o755)
        
        print(f"✅ Deployment scripts oluşturuldu: {scripts_dir}")
        return scripts_dir
    
    def generate_monitoring_config(self):
        """Monitoring configuration"""
        monitoring_config = {
            'monitoring': {
                'enabled': True,
                'drift_detection': {
                    'enabled': True,
                    'method': 'ks_test',
                    'threshold': 0.05,
                    'check_interval_minutes': 60
                },
                'performance_tracking': {
                    'enabled': True,
                    'metrics': ['accuracy', 'precision', 'recall', 'f1_score'],
                    'alert_thresholds': {
                        'accuracy': 0.8,
                        'response_time': 2.0
                    }
                },
                'logging': {
                    'level': 'INFO',
                    'file': 'monitoring/application.log',
                    'max_size_mb': 100,
                    'backup_count': 5
                }
            },
            'ab_testing': {
                'enabled': True,
                'default_traffic_split': 0.5,
                'min_samples_for_analysis': 100
            },
            'alerts': {
                'enabled': True,
                'channels': ['log', 'webhook'],
                'webhook_url': None,
                'email_settings': {
                    'enabled': False,
                    'smtp_server': None,
                    'recipients': []
                }
            }
        }
        
        config_path = self.deployment_dir / "monitoring-config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(monitoring_config, f, default_flow_style=False)
        
        print(f"✅ Monitoring config oluşturuldu: {config_path}")
        return config_path
    
    def generate_all_deployment_files(self):
        """Tüm deployment dosyalarını oluştur"""
        print("🚀 TÜM DEPLOYMENT DOSYALARI OLUŞTURULUYOR")
        print("=" * 50)
        
        created_files = []
        
        # Docker files
        docker_files = self.generate_docker_files()
        created_files.extend(docker_files)
        
        # Kubernetes manifests
        k8s_dir = self.generate_kubernetes_manifests()
        created_files.append(k8s_dir)
        
        # Production requirements
        req_file = self.generate_production_requirements()
        created_files.append(req_file)
        
        # Environment template
        env_file = self.generate_env_template()
        created_files.append(env_file)
        
        # Deployment scripts
        scripts_dir = self.generate_deployment_scripts()
        created_files.append(scripts_dir)
        
        # Monitoring config
        monitoring_config = self.generate_monitoring_config()
        created_files.append(monitoring_config)
        
        # README for deployment
        readme_content = f'''# AutoTicket Classifier Deployment

Deployment dosyaları {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} tarihinde oluşturuldu.

## Dosyalar

- `Dockerfile`: Docker container tanımı
- `docker-compose.yml`: Docker Compose konfigürasyonu
- `kubernetes/`: Kubernetes deployment manifests
- `requirements-production.txt`: Production dependencies
- `.env.template`: Environment variables template
- `scripts/`: Deployment scripts
- `monitoring-config.yaml`: Monitoring konfigürasyonu

## Deployment

### Docker ile:
```bash
cd deployment
./scripts/deploy.sh
```

### Kubernetes ile:
```bash
kubectl apply -f kubernetes/
```

## Monitoring

Monitoring dashboard: http://localhost:5000/monitoring
Health check: http://localhost:5000/health

## Rollback

```bash
./scripts/rollback.sh
```
'''
        
        readme_path = self.deployment_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        created_files.append(readme_path)
        
        print(f"✅ Tüm deployment dosyaları oluşturuldu: {len(created_files)} dosya")
        print(f"📁 Deployment klasörü: {self.deployment_dir}")
        
        return created_files

def create_deployment_package():
    """Deployment package oluştur"""
    manager = DeploymentManager()
    return manager.generate_all_deployment_files()

if __name__ == "__main__":
    create_deployment_package()
