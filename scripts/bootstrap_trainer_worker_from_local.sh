INSTANCE_IP="$1"

ssh -i .keys/lagrangian-training-worker-py.pem ec2-user@$INSTANCE_IP

scp -i .keys/lagrangian-training-worker-py.pem \
  scripts/bootstrap_training_worker.sh \
  ec2-user@$INSTANCE_IP:/tmp/bootstrap_training_worker.sh

ssh -i .keys/lagrangian-training-worker-py.pem ec2-user@$INSTANCE_IP '
sudo docker rm -f lagrangian-training-worker-py || true
set -euo pipefail

AWS_REGION="us-east-1"
IMAGE_URI="676726973653.dkr.ecr.us-east-1.amazonaws.com/lagrangian/remoteworkers:latest"
CONTAINER_NAME="lagrangian-training-worker-py"
WORKER_PORT="8765"
WORKER_SHARED_SECRET="9480301160"

REGISTRY="${IMAGE_URI%%/*}"
if [[ "${REGISTRY}" == *".dkr.ecr."* ]]; then
  ECR_LOGIN="aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${REGISTRY}"
else
  ECR_LOGIN="true"
fi

perl -pe "
s#__ECR_LOGIN__#${ECR_LOGIN}#g;
s#__IMAGE_URI__#${IMAGE_URI}#g;
s#__CONTAINER_NAME__#${CONTAINER_NAME}#g;
s#__WORKER_PORT__#${WORKER_PORT}#g;
s#__WORKER_SHARED_SECRET__#${WORKER_SHARED_SECRET}#g;
" /tmp/bootstrap_training_worker.sh > /tmp/bootstrap_training_worker.rendered.sh

chmod +x /tmp/bootstrap_training_worker.rendered.sh
sudo /tmp/bootstrap_training_worker.rendered.sh
sudo docker logs -f --timestamps --tail=200 lagrangian-training-worker-py
'