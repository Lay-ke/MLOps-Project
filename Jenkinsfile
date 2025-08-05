pipeline {
    agent any

    environment {
        REGISTRY = 'mintah'
        MLFLOW_IMAGE_NAME = 'mlflow-server'
        AIRFLOW_IMAGE_NAME = 'airflow-custom'
        INFERENCE_IMAGE_NAME = 'inference-api'
        GIT_CREDENTIALS_ID = 'git-creds-id'
        DOCKER_CREDENTIALS_ID = 'docker-creds-id'
        INFRA_REPO_URL = 'https://github.com/Lay-ke/MLOps-IAC.git'
        IMAGE_TAG = "${env.BUILD_NUMBER}"
        KUBECONFIG_CREDENTIALS_ID = 'kubeconfig-creds-id'
    }

    stages {
        stage('Checkout Code') {
            steps {
                checkout scm
            }
        }

        stage('Lint & Test') {
            steps {
                sh 'python3 -m venv mlops-venv'
                sh 'source mlops-venv/bin/activate && pip install -r requirements.txt'
                sh 'source mlops-venv/bin/activate && flake8 .'  // Lint
                sh 'source mlops-venv/bin/activate && pytest tests/'  // Run tests
            }
        }

        stage('Build MLflow Docker Image') {
            steps {
                script {
                    docker.withRegistry('https://index.docker.io/v1/', DOCKER_CREDENTIALS_ID) {
                        def mlflowImage = docker.build("${REGISTRY}/${MLFLOW_IMAGE_NAME}:${IMAGE_TAG}", "-f Dockerfile.mlflow .")
                        mlflowImage.push()
                        mlflowImage.push('latest')
                    }
                }
            }
        }

        stage('Build Airflow Docker Image') {
            steps {
                script {
                    docker.withRegistry('https://index.docker.io/v1/', DOCKER_CREDENTIALS_ID) {
                        def airflowImage = docker.build("${REGISTRY}/${AIRFLOW_IMAGE_NAME}:${IMAGE_TAG}", "-f Dockerfile.airflow .")
                        airflowImage.push()
                        airflowImage.push('latest')
                    }
                }
            }
        }

        stage("Build Inference Docker Image") {
            steps {
                dir('inference') {
                    script {
                        docker.withRegistry('https://index.docker.io/v1/', DOCKER_CREDENTIALS_ID) {
                            def inferenceImage = docker.build("${REGISTRY}/${INFERENCE_IMAGE_NAME}:${IMAGE_TAG}", "-f Dockerfile .")
                            inferenceImage.push()
                            inferenceImage.push('latest')
                        }
                    }
                }
            }
        }

        stage('Update Kubernetes Manifests') {
            steps {
                script {
                    // Clone the infrastructure repository
                    dir('infra') {
                        git url: INFRA_REPO_URL, branch: 'main'

                        // Install yq if not available (fallback to sed if needed)
                        // sh '''
                        //     if ! command -v yq &> /dev/null; then
                        //         echo "Installing yq..."
                        //         wget -qO /tmp/yq https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64
                        //         chmod +x /tmp/yq
                        //         sudo mv /tmp/yq /usr/local/bin/yq
                        //     fi
                        // '''

                        // Update MLflow deployment image
                        sh """
                            if [ -f k8s-infra/mlflow/deployment.yaml ]; then
                                echo "Updating MLflow image to ${REGISTRY}/${MLFLOW_IMAGE_NAME}:${IMAGE_TAG}"
                                yq e '(.spec.template.spec.containers[] | select(.name == "mlflow").image) = "${REGISTRY}/${MLFLOW_IMAGE_NAME}:${IMAGE_TAG}"' -i k8s-infra/mlflow/deployment.yaml
                            else
                                echo "MLflow deployment.yaml not found in k8s-infra/mlflow/"
                            fi
                        """

                        // Update Airflow Helm values image
                        sh """
                            if [ -f k8s-infra/airflow/values.yaml ]; then
                                echo "Updating Airflow image to ${REGISTRY}/${AIRFLOW_IMAGE_NAME}:${IMAGE_TAG}"
                                yq e '.images.airflow.repository = "${REGISTRY}/${AIRFLOW_IMAGE_NAME}"' -i k8s-infra/airflow/values.yaml
                                yq e '.images.airflow.tag = "${IMAGE_TAG}"' -i k8s-infra/airflow/values.yaml
                            else
                                echo "Airflow values.yaml not found in k8s-infra/airflow/"
                            fi
                        """

                        // Update Inference API deployment image
                        sh """
                            if [ -f k8s-infra/inference/deployment.yaml ]; then
                                echo "Updating Inference API image to ${REGISTRY}/${INFERENCE_IMAGE_NAME}:${IMAGE_TAG}"
                                yq e '(.spec.template.spec.containers[] | select(.name == "inference-api").image) = "${REGISTRY}/${INFERENCE_IMAGE_NAME}:${IMAGE_TAG}"' -i k8s-infra/inference/deployment.yaml
                            else
                                echo "Inference API deployment.yaml not found in k8s-infra/inference/"
                            fi
                        """

                        // Commit and push changes
                        sh 'git config user.name "jenkins"'
                        sh 'git config user.email "jenkins@ci"'
                        sh 'git add manifests/'
                        sh """git commit -m "Update images: MLflow:${IMAGE_TAG}, Airflow:${IMAGE_TAG}, Inference:${IMAGE_TAG}" || echo "No changes to commit" """
                        sh 'git push origin main'
                    }
                }
            }
        }

        stage('Deploy Secrets') {
            steps {
                script {
                    dir('infra') {
                        // Clone the infrastructure repository if not already done
                        if (!fileExists('k8s-infra')) {
                            git url: INFRA_REPO_URL, branch: 'main'
                        }
                        withCredentials([
                            file(credentialsId: KUBECONFIG_CREDENTIALS_ID, variable: 'KUBECONFIG'),
                            string(credentialsId: 'aws-access-key-id', variable: 'AWS_ACCESS_KEY'),
                            string(credentialsId: 'aws-secret-access-key', variable: 'AWS_SECRET_KEY'),
                            string(credentialsId: 'airflow-db-password', variable: 'AIRFLOW_DB_PASSWORD'),
                            string(credentialsId: 'airflow-web-password', variable: 'AIRFLOW_WEB_PASSWORD')
                        ]) {
                            sh """
                                # Create namespaces if they don't exist
                                kubectl create namespace mlflow --dry-run=client -o yaml | kubectl apply -f -
                                kubectl create namespace inference --dry-run=client -o yaml | kubectl apply -f -
                                kubectl create namespace airflow --dry-run=client -o yaml | kubectl apply -f -

                                # Create secrets for MLflow namespace
                                echo "Creating secrets for MLflow namespace..."
                                kubectl create secret generic aws-credentials \
                                    --from-literal=AWS_ACCESS_KEY_ID='${AWS_ACCESS_KEY}' \
                                    --from-literal=AWS_SECRET_ACCESS_KEY='${AWS_SECRET_KEY}' \
                                    --from-literal=AWS_DEFAULT_REGION='us-east-1' \
                                    --namespace=mlflow \
                                    --dry-run=client -o yaml | kubectl apply -f -

                                kubectl create secret generic mlflow-config \
                                    --from-literal=MLFLOW_TRACKING_URI='http://mlflow-server.mlflow.svc.cluster.local:5000' \
                                    --namespace=mlflow \
                                    --dry-run=client -o yaml | kubectl apply -f -

                                # Create secrets for inference namespace
                                echo "Creating secrets for inference namespace..."
                                kubectl create secret generic aws-credentials \
                                    --from-literal=AWS_ACCESS_KEY_ID='${AWS_ACCESS_KEY}' \
                                    --from-literal=AWS_SECRET_ACCESS_KEY='${AWS_SECRET_KEY}' \
                                    --from-literal=AWS_DEFAULT_REGION='us-east-1' \
                                    --namespace=inference \
                                    --dry-run=client -o yaml | kubectl apply -f -
                                    
                                kubectl create secret generic mlflow-config \
                                    --from-literal=MLFLOW_TRACKING_URI='http://mlflow-server.mlflow.svc.cluster.local:5000' \
                                    --namespace=inference \
                                    --dry-run=client -o yaml | kubectl apply -f -

                                # Create secrets for airflow namespace (if needed)
                                echo "Creating secrets for airflow namespace..."
                                kubectl create secret generic aws-credentials \
                                    --from-literal=AWS_ACCESS_KEY_ID='${AWS_ACCESS_KEY}' \
                                    --from-literal=AWS_SECRET_ACCESS_KEY='${AWS_SECRET_KEY}' \
                                    --from-literal=AWS_DEFAULT_REGION='us-east-1' \
                                    --namespace=airflow \
                                    --dry-run=client -o yaml | kubectl apply -f -
                                    
                                kubectl create secret generic mlflow-config \
                                    --from-literal=MLFLOW_TRACKING_URI='http://mlflow-server.mlflow.svc.cluster.local:5000' \
                                    --from-literal=INFERENCE_API_URL='http://inference-service.inference.svc.cluster.local:8000' \
                                    --namespace=airflow \
                                    --dry-run=client -o yaml | kubectl apply -f -

                                # Create database credentials for Airflow
                                echo "Creating database credentials for airflow namespace..."
                                kubectl create secret generic airflow-db-credentials \
                                    --from-literal=username='airflow' \
                                    --from-literal=password='${AIRFLOW_DB_PASSWORD}' \
                                    --namespace=airflow \
                                    --dry-run=client -o yaml | kubectl apply -f -

                                # Create web credentials for Airflow
                                echo "Creating web credentials for airflow namespace..."
                                kubectl create secret generic airflow-web-credentials \
                                    --from-literal=username='admin' \
                                    --from-literal=password='${AIRFLOW_WEB_PASSWORD}' \
                                    --namespace=airflow \
                                    --dry-run=client -o yaml | kubectl apply -f -

                                echo "Secrets created successfully!"
                            """ 
                        }
                    }
                }
            }
        }

        stage('Deploy to Kubernetes') {
            steps {
                script {
                    // Use kubectl to apply the updated manifests directly (optional - ArgoCD will handle this)
                    withCredentials([file(credentialsId: KUBECONFIG_CREDENTIALS_ID, variable: 'KUBECONFIG')]) {
                        sh """
                            # Apply MLflow manifests
                            if [ -d "k8s-infra/mlflow" ]; then
                                kubectl apply -f k8s-infra/mlflow/
                            fi
                            
                            # Apply inference API manifests
                            if [ -d "k8s-infra/inference" ]; then
                                kubectl apply -f k8s-infra/inference/
                            fi
                            
                            # Update Airflow using Helm with new values
                            if [ -f "k8s-infra/airflow/values.yaml" ]; then
                                helm repo add apache-airflow https://airflow.apache.org || true
                                helm repo update
                                helm upgrade --install airflow apache-airflow/airflow \\
                                    --namespace airflow \\
                                    --values k8s-infra/airflow/values.yaml
                            fi
                            
                            # Wait for deployments to be ready
                            echo "Waiting for deployments to be ready..."
                            kubectl rollout status deployment/mlflow-server -n mlflow --timeout=300s || echo "MLflow deployment not found or failed"
                            kubectl rollout status deployment/airflow-scheduler -n airflow --timeout=300s || echo "Airflow deployment not found or failed"
                            kubectl rollout status deployment/inference-api -n default --timeout=300s || echo "Inference API deployment not found or failed"
                        """
                    }
                }
            }
        }
    }

    post {
        always {
            cleanWs()
        }
    }
}
