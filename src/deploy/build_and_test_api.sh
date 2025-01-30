#!/bin/bash
set -e  # Stoppe l'exécution en cas d'erreur

# Charger les paramètres du fichier params.yaml (DVC)
DOCKER_FILE=$(cat params.yaml | yq '.build_and_test_api.docker_file')
IMAGE_NAME=$(cat params.yaml | yq '.build_and_test_api.image_name')
TAG_NAME=$(cat params.yaml | yq '.build_and_test_api.tag_name')
INPUT_TEST_FEATURES=$(cat params.yaml | yq '.build_and_test_api.input_test_features')
INPUT_TEST_LABELS=$(cat params.yaml | yq '.build_and_test_api.input_test_labels')
TEST_SCRIPT=$(cat params.yaml | yq '.build_and_test_api.test_script')
REGISTRY_NAME=$(cat params.yaml | yq '.build_and_test_api.registry_name')

# 1️⃣ Build de l’image Docker
echo "🛠️ Building Docker image: ${IMAGE_NAME}:${TAG_NAME}"
docker build -f $DOCKER_FILE -t ${IMAGE_NAME}:${TAG_NAME} .


ls $(pwd)/${TEST_SCRIPT}
ls $(pwd)/${INPUT_TEST_FEATURES}
ls $(pwd)/${INPUT_TEST_LABELS}

echo "Démarrer le container"
docker run -d --name api_container -p 8000:8000 \
	-v $(pwd)/${TEST_SCRIPT}:/app/test_api.py \
	-v $(pwd)/${INPUT_TEST_FEATURES}:/app/x_test.csv \
	-v $(pwd)/${INPUT_TEST_LABELS}:/app/y_test.csv \
	${IMAGE_NAME}:${TAG_NAME}

# 2️⃣ Exécuter les tests
echo "🧪 Running API tests..."
docker exec api_container python test_api.py

# Vérifier si le test est réussi
if [ $? -eq 0 ]; then
    echo "✅ Tests passed, pushing image to registry..."
    
    # 3️⃣ Push de l’image sur le registry DockerHub
    docker tag ${IMAGE_NAME}:${TAG_NAME} ${REGISTRY_NAME}/${IMAGE_NAME}:${TAG_NAME}
    docker push ${REGISTRY_NAME}/${IMAGE_NAME}:${TAG_NAME}

    echo "🚀 Image pushed successfully to ${REGISTRY_NAME}/${IMAGE_NAME}:${TAG_NAME}"
else
    echo "❌ Tests failed. Skipping push."
    exit 1
fi

docker stop api_container && docker rm api_container

